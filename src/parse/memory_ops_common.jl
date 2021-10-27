dottosym(x)::Symbol = x

function dottosym(x::Expr)::Symbol
  s1 = dottosym(x.args[1])
  xa2 = x.args[2]
  xa2 isa QuoteNode ? Symbol(s1, "###extractarray###", xa2.value) : Symbol(s1, "###extractarray###", xa2)
end
function extract_array_symbol_from_ref!(ls::LoopSet, ex::Expr, offset1::Int)::Symbol
  ar = ex.args[1 + offset1]
  if isa(ar, Symbol)
    return ar
  elseif isa(ar, Expr) && (ar.head === :(.) || ar.head === :ref)
    s = dottosym(ar)
    # pushpreamble!(ls, Expr(:(=), s, ar))
    pushprepreamble!(ls, Expr(:(=), s, ar))
    return s
  else
    throw("Indexing into the following expression was not recognized: $ar")
  end
end


function ref_from_expr!(ls, ex, offset1::Int, offset2::Int)
    ar = extract_array_symbol_from_ref!(ls, ex, offset1)
    ar, @view(ex.args[2 + offset2:end])
end
ref_from_ref!(ls::LoopSet, ex::Expr) = ref_from_expr!(ls, ex, 0, 0)
ref_from_getindex!(ls::LoopSet, ex::Expr) = ref_from_expr!(ls, ex, 1, 1)
ref_from_setindex!(ls::LoopSet, ex::Expr) = ref_from_expr!(ls, ex, 1, 2)
function ref_from_expr!(ls::LoopSet, ex::Expr)
  if ex.head === :ref
    ref_from_ref!(ls, ex)
  else#if ex.head === :call
    f = first(ex.args)::Symbol
    f === :getindex ? ref_from_getindex!(ls, ex) : ref_from_setindex!(ls, ex)
  end
end

add_vptr!(ls::LoopSet, op::Operation) = add_vptr!(ls, op.ref)
add_vptr!(ls::LoopSet, mref::ArrayReferenceMeta) = add_vptr!(ls, mref.ref.array, vptr(mref))
# using VectorizationBase: noaliasstridedpointer
presbufsym(array) = Symbol('#', array, "#preserve#buffer#")
function add_vptr!(ls::LoopSet, array::Symbol, vptrarray::Symbol, actualarray::Bool = true)
  if !includesarray(ls, array)
    push!(ls.includedarrays, array)
    actualarray && push!(ls.includedactualarrays, array)
    pushpreamble!(ls, Expr(:(=), Expr(:tuple, vptrarray, presbufsym(array)), Expr(:call, lv(:stridedpointer_preserve), array)))
  end
  nothing
end

@inline staticdims(::Any) = One()
@inline staticdims(::CartesianIndices{N}) where {N} = StaticInt{N}()

function append_loop_staticdims!(valcall::Expr, loop::Loop, constoffset::Int)
  isstaticloop(loop) && return constoffset + 1
  push!(valcall.args, Expr(:call, lv(:staticdims), loop.rangesym))
  constoffset
end
function subset_vptr!(ls::LoopSet, vptr::Symbol, indnum::Int, ind, previndices, loopindex, D::Int)
  subset = D == 0
  str_typ = subset ? "subset" : "index"
  subsetvptr = Symbol(vptr, "_##$(str_typ)##_$(indnum)_##with##_$(ind)_##")
  valcall = Expr(:call, +)
  constoffset = 1
  if indnum > 1
    offset = first(previndices) === DISCONTIGUOUS
    # valcall =  Expr(:call, :(+), valcall)
    for i ∈ 1:indnum-1
      if loopindex[i]
        loopdep = previndices[i+offset]
        if loopdep === CONSTANTZEROINDEX
          constoffset += 1
          continue
        end
      else
        # assumes all staticdims will be of equal length once expanded...
        # A[I + J, constindex], I and J may be CartesianIndices. This requires they all be of same number of dims
        loopdep = first(loopdependencies(ls.opdict[previndices[i+offset]]))
      end
      constoffset = append_loop_staticdims!(valcall, getloop(ls, loopdep), constoffset)
    end
  end
  # indm1 = ind isa Integer ? ind - 1 : Expr(:call, :-, ind, 1)
  f = lv(Core.ifelse(subset, :subsetview, :_gesp))
  constoffsetexpr = staticexpr(constoffset)
  if length(valcall.args) ≠ 1
    push!(valcall.args, constoffsetexpr)
    constoffsetexpr = valcall
  end
  fexpr = Expr(:call, f, vptr, constoffsetexpr, ind)
  subset || push!(fexpr.args, staticexpr(D))
  pushpreamble!(ls, Expr(:(=), subsetvptr, fexpr))
  subsetvptr
end

function gesp_const_offset!(ls::LoopSet, vptrarray::Symbol, ninds::Int, indices::Vector{Symbol}, loopedindex::Vector{Bool}, mlt::Integer, sym, D::Int)
  if isone(mlt)
    subset_vptr!(ls, vptrarray, ninds, sym, indices, loopedindex, D)
  else        
    # want same name for arrays to be given the same name if possible
    mltsym = Symbol(sym, "##multiplied##by##", mlt)
    pushpreamble!(ls, Expr(:(=), mltsym, Expr(:call, :(*), mlt, sym)))
    # pushprepreamble!(ls, Expr(:(=), mltsym, Expr(:call, :(*), mlt, sym)))
    subset_vptr!(ls, vptrarray, ninds, mltsym, indices, loopedindex, D)
  end
end
function gesp_const_offsets!(ls::LoopSet, vptrarray::Symbol, ninds::Int, indices::Vector{Symbol}, loopedindex::Vector{Bool}, mltsyms::Vector{Tuple{Int,Symbol}}, D::Int)
  length(mltsyms) > 1 && sort!(mltsyms, by = last) # if multiple have same combination of syms, make sure they match even if order is different
  for (mlt,sym) ∈ mltsyms
    vptrarray = gesp_const_offset!(ls, vptrarray, ninds, indices, loopedindex, mlt, sym, D)
  end
  vptrarray
end


byterepresentable(x)::Bool = false
byterepresentable(x::Integer)::Bool = typemin(Int8) ≤ x ≤ typemax(Int8)
function _addoffset!(indices, offsets, strides, loopedindex, loopdependencies, ind, offset, stride)
    push!(indices, ind)
    push!(offsets, offset % Int8)
    # push!(offsets, (offset+stride-1) % Int8)
    push!(strides, stride % Int8)
    push!(loopedindex, true)
    push!(loopdependencies, ind)
    true
end

function addconstindex!(indices, offsets, strides, loopedindex, offset)
    push!(indices, CONSTANTZEROINDEX)
    # push!(offsets, (offset-1) % Int8)
    push!(offsets, (offset) % Int8)
    push!(strides, zero(Int8))
    push!(loopedindex, true)
    true
end
function addopindex!(
    parents::Vector{Operation}, loopdependencies::Vector{Symbol}, reduceddeps::Vector{Symbol},
    indices::Vector{Symbol}, offsets::Vector{Int8}, strides::Vector{Int8}, loopedindex::Vector{Bool}, indop::Operation,
    stride = one(Int8), offset = zero(Int8)
)
    pushparent!(parents, loopdependencies, reduceddeps, indop)
    push!(indices, name(indop));
    push!(offsets, offset)
    push!(strides, stride)
    push!(loopedindex, false)
    nothing
end

####################################################################################################
###################################### Index Parsing Helpers #######################################
####################################################################################################

function add_affine_index_expr!(ls::LoopSet, mult_syms::Vector{Tuple{Int,Symbol}}, constant::Base.RefValue{Int}, stride::Int, expr::Symbol)
    push!(mult_syms, (stride, expr))
    return nothing
end
function add_affine_index_expr!(ls::LoopSet, mult_syms::Vector{Tuple{Int,Symbol}}, constant::Base.RefValue{Int}, stride::Int, expr::Integer)
    constant[] += stride*expr
    return nothing
end
function add_affine_op!(ls::LoopSet, mult_syms::Vector{Tuple{Int,Symbol}}, constant::Base.RefValue{Int}, stride::Int, expr::Expr)
    parent = add_operation!(ls, gensym!(ls, "indexpr"), expr, sizeof(Int), length(ls.loopsymbols))
    add_affine_index_expr!(ls, mult_syms, constant, stride, name(parent))    
    return nothing
end
function add_mul!(ls::LoopSet, mult_syms::Vector{Tuple{Int,Symbol}}, constant::Base.RefValue{Int}, stride::Int, arg1, arg2, expr)
    if arg1 isa Integer
        add_affine_index_expr!(ls, mult_syms, constant, stride * arg1, arg2)
    elseif arg2 isa Integer
        add_affine_index_expr!(ls, mult_syms, constant, stride * arg2, arg1)
    else
        add_affine_op!(ls, mult_syms, constant, stride, expr)
    end
    return nothing
end
function add_affine_index_expr!(ls::LoopSet, mult_syms::Vector{Tuple{Int,Symbol}}, constant::Base.RefValue{Int}, stride::Int, expr::Expr)
    expr.head === :call || return add_affine_op!(ls, mult_syms, constant, stride, expr)
    f = expr.args[1]
    if f === :(*)
        @assert length(expr.args) == 3
        add_mul!(ls, mult_syms, constant, stride, expr.args[2], expr.args[3], expr)
    elseif f === :(-)
        if length(expr.args) == 3
            add_affine_index_expr!(ls, mult_syms, constant, stride, expr.args[2])
        elseif length(expr.args) ≠ 2
            throw("Subtraction expression: $expr has an invalid number of arguments.")            
        end
        add_affine_index_expr!(ls, mult_syms, constant, -stride, last(expr.args))
    elseif f === :(+)
        for i ∈ 2:length(expr.args)
            add_affine_index_expr!(ls, mult_syms, constant, stride, expr.args[i])
        end
    else
        add_affine_op!(ls, mult_syms, constant, stride, expr)
    end
    return nothing
end
function affine_index_expression(ls::LoopSet, expr)::Tuple{Int,Vector{Tuple{Int,Symbol}}}
    mult_syms = Tuple{Int,Symbol}[]
    constant = Ref(0)
    add_affine_index_expr!(ls, mult_syms, constant, 1, expr)
    return constant[], mult_syms
end

function muladd_index!(
    ls::LoopSet, parents, loopdependencies, reduceddeps, indices, offsets, strides, loopedindex,
    mlt::Int, sym::Symbol, offset::Int
)
    muladd_index!(ls, parents, loopdependencies, reduceddeps, indices, offsets, strides, loopedindex, mlt, getop(ls, sym, sizeof(Int)), offset)
end
function muladd_op!(ls::LoopSet, mlt::Int, sym::Symbol, offset::Int)
    muladd_op!(ls, mlt, getop(ls, sym, sizeof(Int)), offset)
end
function muladd_op!(ls::LoopSet, mlt::Int, symop::Operation, offset::Int)
    indsym = gensym!(ls, "affindexop")
    vparents = [symop]
    # vparents = [, mltop, offop]
    if mlt == -1
        f = :vsub_nsw
        if offset ≠ 0
            pushfirst!(vparents, add_constant!(ls, offset, sizeof(Int)))
        end
    elseif mlt == 1
        offset == 0 && return only(vparents)
        push!(vparents, add_constant!(ls, offset, sizeof(Int)))
        f = :vadd_nsw
    else
        push!(vparents, add_constant!(ls, mlt, sizeof(Int)))
        f = if offset == 0
            :vmul_nsw
        else            
            push!(vparents, add_constant!(ls, offset, sizeof(Int)))
            :muladd
        end
    end
    add_compute!(ls, indsym, instruction(f), vparents, sizeof(Int))
end
function muladd_index!(
    ls::LoopSet, parents, loopdependencies, reduceddeps, indices, offsets, strides, loopedindex,
    mlt::Int, symop::Operation, offset::Int
)
    if byterepresentable(offset) & byterepresentable(mlt)
        addopindex!(
            parents, loopdependencies, reduceddeps, indices,
            offsets, strides, loopedindex, symop, mlt, offset
        )
    else        
        indop = muladd_op!(ls, mlt, symop, offset)
        addopindex!(parents, loopdependencies, reduceddeps, indices, offsets, strides, loopedindex, indop)
    end
end

function getintconstant(ls::LoopSet, op::Operation)
  for (i, (v,bits,bool)) ∈ ls.preamble_symint
    i == identifier(op) || continue
    return byterepresentable(v), v%Int8
  end
  return false, zero(Int8)
end

function add_additive_index!(ls::LoopSet, opparents, vptrarray, sym, mop, ninds, indices, offsets, strides, loopedindex, loopdependencies, reduceddeps, offset, mlt, D)
  factor = Core.ifelse((instruction(mop).instr === :sub_fast), -1, 1)
  if length(parents(mop)) == 2
    sub1 = parents(mop)[1]
    sub2 = parents(mop)[2]
    if isloopvalue(sub1) & isconstant(sub2)
      isliteralint, literalval = getintconstant(ls, sub2)
      if isliteralint
        _addoffset!(indices, offsets, strides, loopedindex, loopdependencies, name(sub1), offset+literalval, mlt)
      else
        vptrarray = gesp_const_offset!(ls, vptrarray, ninds, indices, loopedindex, factor*mlt, name(sub2), D)
        _addoffset!(indices, offsets, strides, loopedindex, loopdependencies, name(sub1), offset, mlt)
      end
    elseif isloopvalue(sub2) & isconstant(sub1)
      isliteralint, literalval = getintconstant(ls, sub1)
      if isliteralint
        _addoffset!(indices, offsets, strides, loopedindex, loopdependencies, name(sub2), offset+literalval, factor*mlt)
      else
        vptrarray = gesp_const_offset!(ls, vptrarray, ninds, indices, loopedindex, mlt, name(sub1), D)
        _addoffset!(indices, offsets, strides, loopedindex, loopdependencies, name(sub2), offset, factor*mlt)
      end
    else
      muladd_index!(ls, opparents, loopdependencies, reduceddeps, indices, offsets, strides, loopedindex, mlt, sym, offset)
    end
  else
    muladd_index!(ls, opparents, loopdependencies, reduceddeps, indices, offsets, strides, loopedindex, mlt, sym, offset)
  end
  vptrarray
end

function checkforoffset!(
    ls::LoopSet, vptrarray::Symbol, ninds::Int, opparents::Vector{Operation}, indices::Vector{Symbol}, offsets::Vector{Int8}, strides::Vector{Int8},
    loopedindex::Vector{Bool}, loopdependencies::Vector{Symbol}, reduceddeps::Vector{Symbol}, ind::Expr, D::Int
)::Symbol

  offset, mult_syms = affine_index_expression(ls, ind)
  let deleted = 0, N = length(mult_syms)
    for n ∈ 1:N
      ntemp = n - deleted
      mlt, sym = mult_syms[ntemp]
      opm = get(ls.opdict, sym, nothing)
      opm === nothing && continue
      isconstant(opm) || continue
      found = false
      for (opid,(intval,intsz,signed)) ∈ ls.preamble_symint
        if opid == identifier(opm)
          offset += intval * mlt
          deleted += 1
          deleteat!(mult_syms, ntemp)
          found = true
          break
        end
      end
      found && continue
      for (opid,nt) ∈ ls.preamble_zeros
        if opid == identifier(opm)
          deleted += 1
          deleteat!(mult_syms, ntemp)
          break
        end
      end
    end
  end
  if !byterepresentable(offset)
    if length(mult_syms) == 1
      mlt,sym = only(mult_syms)
      if !byterepresentable(mlt)
        # this is so we don't unnecessarilly add a separate offset
        muladd_index!(ls, opparents, loopdependencies, reduceddeps, indices, offsets, strides, loopedindex, mlt, sym, offset)
        return vptrarray
      end
    end
    r = copysign(abs(offset) & 127, offset)
    vptrarray = gesp_const_offset!(ls, vptrarray, ninds, indices, loopedindex, 1, offset - r, D)
    offset = r
  end
  # (success && byterepresentable(offset)) || return false, vptrarray
  if length(mult_syms) == 0
    addconstindex!(indices, offsets, strides, loopedindex, offset)
    return vptrarray
  elseif length(mult_syms) == 1
    mlt, sym = mult_syms[1]
    if sym ∈ ls.loopsymbols
      if byterepresentable(mlt)
        _addoffset!(indices, offsets, strides, loopedindex, loopdependencies, sym, offset, mlt)
      else
        muladd_index!(ls, opparents, loopdependencies, reduceddeps, indices, offsets, strides, loopedindex, mlt, sym, offset)
      end
    elseif !byterepresentable(mlt)
      muladd_index!(ls, opparents, loopdependencies, reduceddeps, indices, offsets, strides, loopedindex, mlt, sym, offset)
    else
      mop = get(ls.opdict, sym, nothing)
      if mop === nothing
        vptrarray = gesp_const_offset!(ls, vptrarray, ninds, indices, loopedindex, mlt, sym, D)
        addconstindex!(indices, offsets, strides, loopedindex, offset)
      elseif isconstant(mop)
        vptrarray = gesp_const_offset!(ls, vptrarray, ninds, indices, loopedindex, mlt, name(mop), D)
        addconstindex!(indices, offsets, strides, loopedindex, offset)
      elseif (instruction(mop).instr === :add_fast) || (instruction(mop).instr === :sub_fast)
        vptrarray = add_additive_index!(ls, opparents, vptrarray, sym, mop, ninds, indices, offsets, strides, loopedindex, loopdependencies, reduceddeps, offset, mlt, D)
      else
        muladd_index!(ls, opparents, loopdependencies, reduceddeps, indices, offsets, strides, loopedindex, mlt, sym, offset)
      end
    end
    return vptrarray
  end
  loopsym_ind = 0
  operations = Operation[]
  operation_mults = Int[]
  deleteat_inds = Int[]
  for (i,(m,s)) ∈ enumerate(mult_syms)
    if s ∈ ls.loopsymbols
      if loopsym_ind == 0
        loopsym_ind = i
        push!(deleteat_inds, loopsym_ind)
      else
        push!(operations, add_loopvalue!(ls, s, sizeof(Int)))
        push!(deleteat_inds, i)
        push!(operation_mults, m)
      end
    else
      sop = get(ls.opdict, s, nothing)
      if sop === nothing
        push!(deleteat_inds, i)
        vptrarray = gesp_const_offset!(ls, vptrarray, ninds, indices, loopedindex, m, s, D)
      elseif isconstant(sop)
        push!(deleteat_inds, i)
        vptrarray = gesp_const_offset!(ls, vptrarray, ninds, indices, loopedindex, m, name(sop), D)
      else
        # @show sop
        # if instruction(sop).instr === :sub_fast
        #   if length(parents(sop)) == 1
        #     sub1 = parents(sop)[1]
        #   elseif length(parents(op)) == 2
        #     sub1 = parents(sop)[1]
        #     sub2 = parents(sop)[2]
        #     if isloopvalue(sub1)
        #     elseif isloopvalue(sub2)
        #       @show sub2
        #     end
        #   end
        # elseif instruction(op).instr === :add_fast
        # end
        push!(operations, getop(ls, s, 8))
        push!(operation_mults, m)
        push!(deleteat_inds, i)
      end
    end
  end
  if (length(operations) > 0) & (loopsym_ind > 0) # turn into an operation
    _m,_s = mult_syms[loopsym_ind]
    push!(operations, add_loopvalue!(ls, _s, sizeof(Int)))
    push!(operation_mults, _m)
  end
  if length(operations) == 0
    if loopsym_ind == 0
      addconstindex!(indices, offsets, strides, loopedindex, offset)
    else
      mlt, sym = mult_syms[loopsym_ind]
      _addoffset!(indices, offsets, strides, loopedindex, loopdependencies, sym, offset, mlt)
    end
    deleteat!(mult_syms, deleteat_inds)
    return gesp_const_offsets!(ls, vptrarray, ninds, indices, loopedindex, mult_syms, D)
  end
  deleteat!(mult_syms, deleteat_inds)
  vptrarray = gesp_const_offsets!(ls, vptrarray, ninds, indices, loopedindex, mult_syms, D)
  if length(operations) == 1
    _mlt = only(operation_mults)
    indop = muladd_op!(ls, Core.ifelse(byterepresentable(_mlt), 1, _mlt), only(operations), 0)
    addopindex!(opparents, loopdependencies, reduceddeps, indices, offsets, strides, loopedindex, indop, Core.ifelse(byterepresentable(_mlt), _mlt%Int8, one(Int8)), offset%Int8)
  else
    mlt1ind = findfirst(isone, operation_mults)
    opbase = if mlt1ind === nothing # if none of them have a multiplier of `1`, pick the base arbtirarility
      muladd_op!(ls, pop!(operation_mults), pop!(operations), 0)
    else # otherwise, we start accumulating with a 1-multiplier.
      deleteat!(operation_mults, mlt1ind)
      popat!(operations, mlt1ind)
    end
    for i ∈ eachindex(operations)
      _op = operations[i]
      _mlt = operation_mults[i]
      opbase = if _mlt == -1
        add_compute!(ls, gensym!(ls, "indexaccum"), instruction(:(-)), [opbase, _op], sizeof(Int))
      elseif _mlt == 1
        add_compute!(ls, gensym!(ls, "indexaccum"), instruction(:(+)), [opbase, _op], sizeof(Int))
      else
        add_compute!(ls, gensym!(ls, "indexaccum"), instruction(:muladd), [add_constant!(ls, _mlt, sizeof(Int)), _op, opbase], sizeof(Int))
      end
    end
    addopindex!(opparents, loopdependencies, reduceddeps, indices, offsets, strides, loopedindex, opbase, one(Int8), offset%Int8)
  end
  return vptrarray
end

function move_to_last!(x, i)
    i == length(x) && return
    xᵢ = x[i]
    deleteat!(x, i)
    push!(x, xᵢ)
    nothing
end
# TODO: Make this work with Cartesian Indices
function repeated_index!(ls::LoopSet, indices::Vector{Symbol}, vptr::Symbol, indnum::Int, firstind::Int)
    # Move ind to last position
    vptrrepremoved = Symbol(vptr, "##ind##", firstind, "##repeated##", indnum, "##")
    f = Expr(:(.), Expr(:(.), :LoopVectorization, QuoteNode(:VectorizationBase)), QuoteNode(:double_index))
    fiv = Expr(:call, Expr(:curly, :Val, firstind - 1))
    siv = Expr(:call, Expr(:curly, :Val, indnum - 1))
    pushpreamble!(ls, Expr(:(=), vptrrepremoved, Expr(:call, f, vptr, fiv, siv)))
    vptrrepremoved
end

function array_reference_meta!(ls::LoopSet, array::Symbol, rawindices, elementbytes::Int, var::Union{Nothing,Symbol} = nothing)
  vptrarray = vptr(array)
  add_vptr!(ls, array, vptrarray) # now, subset
  indices = Symbol[]
  offsets = Int8[]
  strides = Int8[]
  loopedindex = Bool[]
  parents = Operation[]
  loopdependencies = Symbol[]
  reduceddeps = Symbol[]
  loopset = ls.loopsymbols
  ninds = 1
  D = length(rawindices)
  for ind ∈ rawindices
    if ind isa Integer # subset
      if byterepresentable(ind)
        addconstindex!(indices, offsets, strides, loopedindex, ind)
        ninds += 1
      else
        vptrarray = subset_vptr!(ls, vptrarray, ninds, ind, indices, loopedindex, 0)
        length(indices) == 0 && push!(indices, DISCONTIGUOUS)
      end
    elseif ind isa Expr
      #FIXME: position (in loopnest) wont be length(ls.loopsymbols) in general
      vptrarray = checkforoffset!(
        ls, vptrarray, ninds, parents, indices, offsets, strides, loopedindex, loopdependencies, reduceddeps, ind, D
      )
      ninds += 1
    elseif ind isa Symbol
      if ind ∈ loopset
        ind_prev_index = findfirst(Base.Fix2(===,ind), indices)
        if ind_prev_index === nothing
          push!(indices, ind); ninds += 1
          push!(offsets, zero(Int8))
          push!(strides, one(Int8))
          push!(loopedindex, true)
          push!(loopdependencies, ind)
        else
          move_to_last!(indices, ind_prev_index)
          move_to_last!(offsets, ind_prev_index)
          move_to_last!(strides, ind_prev_index)
          move_to_last!(loopedindex, ind_prev_index)
          move_to_last!(loopdependencies, ind_prev_index)
          vptrarray = repeated_index!(ls, indices, vptrarray, ninds, ind_prev_index + (first(indices) === DISCONTIGUOUS))
          makediscontiguous!(indices)
        end
      else
        indop = get(ls.opdict, ind, nothing)
        if indop !== nothing  && !isconstant(indop)
          if iscompute(indop) && Base.sym_in(instruction(indop).instr, (:add_fast, :sub_fast))
            vptrarray = add_additive_index!(ls, parents, vptrarray, ind, indop, ninds, indices, offsets, strides, loopedindex, loopdependencies, reduceddeps, 0, 1, D)
          else
            pushparent!(parents, loopdependencies, reduceddeps, indop)
            push!(indices, name(indop)); ninds += 1
            push!(offsets, zero(Int8))
            push!(strides, one(Int8))
            push!(loopedindex, false)
          end
        else
          vptrarray = subset_vptr!(ls, vptrarray, ninds, ind, indices, loopedindex, 0)
          length(indices) == 0 && push!(indices, DISCONTIGUOUS)
        end
      end
    else
      throw("Unrecognized loop index: $ind.")
    end
  end
  mref = ArrayReferenceMeta(ArrayReference( array, indices, offsets, strides ), loopedindex, vptrarray)
  ArrayReferenceMetaPosition(mref, parents, loopdependencies, reduceddeps, var === nothing ? Symbol("") : var )
end
function tryrefconvert(ls::LoopSet, ex::Expr, elementbytes::Int, var::Union{Nothing,Symbol} = nothing)::Tuple{Bool,ArrayReferenceMetaPosition}
    ya, yinds = if ex.head === :ref
        ref_from_ref!(ls, ex)
    elseif ex.head === :call
        f = first(ex.args)
        if f === :getindex
            ref_from_getindex!(ls, ex)
        elseif f === :setindex!
            ref_from_setindex!(ls, ex)
        else
            return false, NOTAREFERENCEMP
        end
    else
        return false, NOTAREFERENCEMP
    end
    true, array_reference_meta!(ls, ya, yinds, elementbytes, var)
end
