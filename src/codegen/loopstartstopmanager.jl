function uniquearrayrefs(ls::LoopSet)
  uniquerefs = ArrayReferenceMeta[]
  # each `Vector{Tuple{Int,Int}}` has the same name
  # 1 to ≥1 maps:
  name_to_array_map = Vector{Int}[] # refname -> uniquerefs
  unique_to_name_and_op_map = Vector{Tuple{Int,Int,Int}}[] # uniquerefs -> (ind in name_to_array, ind in that vector, ind in operations)
  for (i,op) ∈ enumerate(operations(ls))
    arrayref = op.ref
    arrayref === NOTAREFERENCE && continue
    unique = true
    for (j,namev) ∈ enumerate(name_to_array_map)
      vptr(uniquerefs[first(namev)]) === vptr(arrayref) || continue # no unique
      for (k,unique_id) ∈ enumerate(namev)
        if sameref(uniquerefs[unique_id], arrayref)
          # matching name, matching ref
          push!(unique_to_name_and_op_map[unique_id], (j, k, i))
          unique = false
          break
        end
      end
      if unique # matching name, no matching ref
        push!(uniquerefs, arrayref)
        push!(namev, length(uniquerefs))
        push!(unique_to_name_and_op_map, Tuple{Int,Int,Int}[(j,length(namev), i)])
        unique = false
      end
      break
    end
    if unique # no matching name, no matching ref
      push!(uniquerefs, arrayref)
      push!(name_to_array_map, Int[length(uniquerefs)])
      push!(unique_to_name_and_op_map, Tuple{Int,Int,Int}[( length(name_to_array_map), 1, i)])
    end
  end
  uniquerefs, name_to_array_map, unique_to_name_and_op_map
end

otherindexunrolled(loopsym::Symbol, ind::Symbol, loopdeps::Vector{Symbol}) = (loopsym !== ind) && (loopsym ∈ loopdeps)
function otherindexunrolled(ls::LoopSet, ind::Symbol, ref::ArrayReferenceMeta)
    us = ls.unrollspecification
    u₁sym = names(ls)[us.u₁loopnum]
    u₂sym = us.u₂loopnum > 0 ? names(ls)[us.u₂loopnum] : Symbol("##undefined##")
    otherindexunrolled(u₁sym, ind, loopdependencies(ref)) || otherindexunrolled(u₂sym, ind, loopdependencies(ref))
end
function multiple_with_name(n::Symbol, v::Vector{ArrayReferenceMeta})
  found = false
  for ref ∈ v
    tst = vptr(ref) === n
    (found & tst) && return true
    found |= tst
  end
  false
end
# multiple_with_name(n::Symbol, v::Vector{ArrayReferenceMeta}) = sum(ref -> n === vptr(ref), v) > 1
# TODO: DRY between indices_calculated_by_pointer_offsets and use_loop_induct_var
function indices_calculated_by_pointer_offsets(ls::LoopSet, ar::ArrayReferenceMeta)
    indices = getindices(ar)
    ls.isbroadcast && return fill(false, length(indices))
    looporder = names(ls)
    offset = isdiscontiguous(ar)
    gespinds = Expr(:tuple)
    out = Vector{Bool}(undef, length(indices))
    li = ar.loopedindex
    # @show ls.vector_width
    for i ∈ eachindex(li)
        ii = i + offset
        ind = indices[ii]
        if (!li[i]) || (ind === CONSTANTZEROINDEX) || multiple_with_name(vptr(ar), ls.lssm.uniquearrayrefs) ||
            (iszero(ls.vector_width) && isstaticloop(getloop(ls, ind)))# ||
            out[i] = false
        elseif (isone(ii) && (first(looporder) === ind))
            out[i] = otherindexunrolled(ls, ind, ar)
        else
            out[i] = true
        end
    end
    out
end

@generated function set_first_stride(sptr::StridedPointer{T,N,C,B,R}) where {T,N,C,B,R}
    minrank = argmin(R)
    newC = C > 0 ? (C == minrank ? 1 : 0) : C
    newB = C > 0 ? (C == minrank ? B : 0) : B #TODO: confirm correctness
    quote
        $(Expr(:meta,:inline))
        # VectorizationBase.StridedPointer{$T,1,$newC,$newB,$(R[minrank],)}($(lv(llvmptr))(sptr), (sptr.strd[$minrank],), (Zero(),))
        VectorizationBase.StridedPointer{$T,1,$newC,$newB,$(R[minrank],)}(pointer(sptr), (sptr.strd[$minrank],), (Zero(),))
    end
end
set_first_stride(x) = x # cross fingers that this works
@inline onetozeroindexgephack(sptr::AbstractStridedPointer) = gesp(set_first_stride(sptr), (Static{-1}(),)) # go backwords
@inline onetozeroindexgephack(sptr::AbstractStridedPointer{T,1}) where {T} = sptr
# @inline onetozeroindexgephack(sptr::StridedPointer{T,1}) where {T} = sptr
@inline onetozeroindexgephack(x) = x

# # Removes parent/child relationship for all children with ref `ar`
# function freechildren!(op::Operation, ar::ArrayReferenceMeta)
#   newchildren = Operation[]
#   for opc ∈ children(op)
#     if opc.ref === ar
#       for (i,opp) ∈ enumerate(parents(opc))
#         if opp === op
#           deleteat!(parents(opc), i)
#           break
#         end
#       end
#     else
#       push!(newchildren, opc)
#     end
#   end
#   op.children = newchildren
#   nothing
# end
# function replaceparent!(ops::Vector{Operation}, )

# end
function set_ref_loopedindex_and_ind!(ref::ArrayReferenceMeta, i::Int, ii::Int, li::Bool, ind::Symbol)
    ref.loopedindex[i] = li
    getindices(ref)[ii] = ind
end
function set_all_to_constant_index!(
  ls::LoopSet, i::Int, ii::Int, indop::Operation, allarrayrefs::Vector{ArrayReferenceMeta},
  array_refs_with_same_name::Vector{Int}, arrayref_to_name_op_collection::Vector{Vector{Tuple{Int,Int,Int}}}
)
  ops = operations(ls)
  for j ∈ array_refs_with_same_name
    arrayref_to_name_op = arrayref_to_name_op_collection[j]
    set_ref_loopedindex_and_ind!(allarrayrefs[j], i, ii, true, CONSTANTZEROINDEX)
    for (_,__,opid) ∈ arrayref_to_name_op
      op = ops[opid]
      set_ref_loopedindex_and_ind!(op.ref, i, ii, true, CONSTANTZEROINDEX)
      delete_ind = 0
      for (k,opp) ∈ enumerate(parents(op))
        if opp === indop
          delete_ind = k
        end
      end
      # @assert delete_ind ≠ 0
      deleteat!(parents(op), delete_ind)
    end
  end
end
maybeloopvaluename(op::Operation) = isloopvalue(op) ? instruction(op).instr : name(op)
function substitute_ops_all!(
  ls::LoopSet, i::Int, ii::Int, indop::Operation, new_parent::Operation, allarrayrefs::Vector{ArrayReferenceMeta},
  array_refs_with_same_name::Vector{Int}, arrayref_to_name_op_collection::Vector{Vector{Tuple{Int,Int,Int}}}
)
  newindsym = maybeloopvaluename(new_parent)
  isloopval = isloopvalue(new_parent)
  ops = operations(ls)
  for j ∈ array_refs_with_same_name
    arrayref_to_name_op = arrayref_to_name_op_collection[j]
    set_ref_loopedindex_and_ind!(allarrayrefs[j], i, ii, isloopval, newindsym)
    for (_,__,opid) ∈ arrayref_to_name_op
      op = ops[opid]
      set_ref_loopedindex_and_ind!(op.ref, i, ii, isloopval, newindsym)
      sub_ind = 0
      for (k,opp) ∈ enumerate(parents(op))
        if opp === indop
          sub_ind = k
        end
      end
      # @assert sub_ind ≠ 0
      if isloopval
        deleteat!(parents(op), sub_ind)
      else
        parents(op)[sub_ind] = new_parent
      end
    end
  end
end

function cse_constant_offsets!(
  ls::LoopSet, q::Expr, ar::ArrayReferenceMeta, allarrayrefs::Vector{ArrayReferenceMeta}, allarrayrefsind::Int,
  array_refs_with_same_name::Vector{Int}, arrayref_to_name_op_collection::Vector{Vector{Tuple{Int,Int,Int}}}
)::Vector{Symbol}
  
  vptrar = vptr(ar)
  arrayref_to_name_op = arrayref_to_name_op_collection[allarrayrefsind]
  us = ls.unrollspecification
  li = ar.loopedindex
  indices = getindices(ar)
  strides = getstrides(ar)
  offset = first(indices) === DISCONTIGUOUS
  gespindoffsets = fill(Symbol(""), length(li))
  for i ∈ eachindex(li)
    li[i] && continue
    ii = i + offset
    ind = indices[ii]
    gespsymbol::Symbol = Symbol("")    
    # we try to licm and offset so we can set `li[i] = true`
    licmoffset = true
    position_in_array_refs_with_same_name = first(arrayref_to_name_op)[2]
    # if position_in_array_refs_with_same_name ≠ 1, then we already performed the op substitutions
    if length(array_refs_with_same_name) > 1 # if it == 1, then there's only 1
      for j ∈ array_refs_with_same_name
        j == position_in_array_refs_with_same_name && continue
        ref = allarrayrefs[j]
        refinds = getindices(ref)
        # refinds === indices && continue # fast check, should be covered by `j == position_in_array_refs_with_same_name`
        if !((refinds[ii] === ind) & (getstrides(ar)[i] == getstrides(ref)[i]))
          # For now, we'll only bother with `licm` if all share the same indices
          # This is so that we can apply the same `licm` to each and keep the same array name.
          # Otherwise, we'll rely on LLVM to optimize indexing.
          licmoffset = false
          break
        end
      end
    end
    licmoffset || continue
    ops = operations(ls)
    # indices are all the same across operations, so we look to the first for checking compatibility...
    memop = indop = ops[first(arrayref_to_name_op)[3]] # indop is a dummy placeholder
    for opp ∈ parents(memop)
      if name(opp) === ind
        indop = opp
      end
    end
    # (we found a match) # iscompute(indop) should be guaranteed...
    ((indop ≢ memop) && iscompute(indop)) || continue
    instr = instruction(indop).instr
    parents_indop = parents(indop)
    if instr === :(+)
      constants_to_licm = Expr(:call, GlobalRef(Base,:(+)))
      new_parent = indop # dummy, for now we will only licm if it lets us remove `indop`
      for opp ∈ parents_indop
        if isconstantop(opp)
          push!(constants_to_licm.args, constantopname(opp))
        elseif new_parent === indop # first
          new_parent = opp
        else # this is the second parent, we give up on `licm`
          licmoffset = false
          break
        end
      end
      licmoffset || continue
      if length(constants_to_licm.args) > 2
        gespindoffsets[i] = gespsymbol = gensym!(ls, "#gespsym#")
        push!(q.args, Expr(:(=), gespsymbol, constants_to_licm))
      elseif length(constants_to_licm.args) == 2
        gespindoffsets[i] = (constants_to_licm.args[2])::Symbol
      else
        licmoffset = false
      end
      if licmoffset
        if new_parent === indop # no parents left
          ind = CONSTANTZEROINDEX
          set_all_to_constant_index!(ls, i, ii, indop, allarrayrefs, array_refs_with_same_name, arrayref_to_name_op_collection)
        else # new_parent is a new parent to replace `indop`
          ind = maybeloopvaluename(new_parent)
          substitute_ops_all!(ls, i, ii, indop, new_parent, allarrayrefs, array_refs_with_same_name, arrayref_to_name_op_collection)
        end
      end
    elseif (instr === :(-)) && length(parents_indop) == 2
      op1 = parents(indop)[1]
      op2 = parents(indop)[2]
      op1const = isconstantop(op1)
      op2const = isconstantop(op2)
      if op1const # op1 - op2
        if op2const
          gespindoffsets[i] = gespsymbol = gensym!(ls, "#gespsym#")
          push!(q.args, Expr(:(=), gespsymbol, Expr(:call, GlobalRef(Base,:(-)), constopname(op1), constopname(op2))))
          ind = CONSTANTZEROINDEX
          set_all_to_constant_index!(ls, i, ii, indop, allarrayrefs, array_refs_with_same_name, arrayref_to_name_op_collection)
        else# op1const, op2dynamic
          # won't bother with this for now
          licmoffset = false
        end
      elseif op2const #op1dynamic
        gespindoffsets[i] = gespsymbol = gensym!(ls, "#gespsym#")
        push!(q.args, Expr(:(=), gespsymbol, Expr(:call, GlobalRef(Base,:(-)), constopname(op2))))
        ind = maybeloopvaluename(op1)
        substitute_ops_all!(ls, i, ii, indop, op1, allarrayrefs, array_refs_with_same_name, arrayref_to_name_op_collection)
      else
        licmoffset = false
      end
    else
      licmoffset = false
    end
  end
  return gespindoffsets    
end

"""
Returns a vector of length equal to the number of indices.
A value > 0 indicates which loop number that index corresponds to when incrementing the pointer.
A value < 0 indicates that abs(value) is the corresponding loop, and a `loopvalue` will be used.
"""
function use_loop_induct_var!(
  ls::LoopSet, q::Expr, ar::ArrayReferenceMeta, allarrayrefs::Vector{ArrayReferenceMeta}, allarrayrefsind::Int,
  array_refs_with_same_name::Vector{Int}, arrayref_to_name_op_collection::Vector{Vector{Tuple{Int,Int,Int}}}
)::Vector{Int}
  us = ls.unrollspecification
  li = ar.loopedindex
  looporder = reversenames(ls)
  uliv = Vector{Int}(undef, length(li))
  indices = getindices(ar)
  offset = first(indices) === DISCONTIGUOUS
  if length(indices) != offset + length(li)
    println(ar)
    throw("Length of indices and length of offset do not match!")
  end
  isbroadcast = ls.isbroadcast
  # no constant offsets when broadcasting
  if (!isbroadcast) && !(all(li))
    checkforgespind = true
    gespindoffsets = cse_constant_offsets!(ls, q, ar, allarrayrefs, allarrayrefsind, array_refs_with_same_name, arrayref_to_name_op_collection)
  else
    checkforgespind = false
    gespindoffsets = indices#dummy
  end
  gespinds = Expr(:tuple)
  offsetprecalc_descript = Expr(:tuple)
  use_offsetprecalc = false
  vptrar = vptr(ar)
  # @show ar
  for i ∈ eachindex(li)
    ii = i + offset
    ind = indices[ii]
    gespsymbol = gespindoffsets[i]
    usegespsymbol = checkforgespind && (gespsymbol ≢ Symbol(""))
    if !li[i] # if it wasn't set
      uliv[i] = 0
      if usegespsymbol
        push!(gespinds.args, gespsymbol)
      else
        push!(gespinds.args, staticexpr(0))
      end
      push!(offsetprecalc_descript.args, 0)
    elseif ind === CONSTANTZEROINDEX
      uliv[i] = 0
      if usegespsymbol
        push!(gespinds.args, Expr(:call, GlobalRef(Base,:(+)), gespsymbol, staticexpr(1)))
      else
        push!(gespinds.args, staticexpr(1))
      end
      push!(offsetprecalc_descript.args, 0)
    elseif isbroadcast ||
      ((isone(ii) && (last(looporder) === ind)) && !(otherindexunrolled(ls, ind, ar)) ||
      multiple_with_name(vptrar, allarrayrefs)) ||
      (iszero(ls.vector_width) && isstaticloop(getloop(ls, ind)))# ||
      # Not doing normal offset indexing
      uliv[i] = -findfirst(Base.Fix2(===,ind), looporder)::Int
      # push!(gespinds.args, Expr(:call, lv(:Zero)))
      if usegespsymbol
        push!(gespinds.args, Expr(:call, GlobalRef(Base,:(+)), gespsymbol, staticexpr(1)))
      else
        push!(gespinds.args, staticexpr(1))
      end
      push!(offsetprecalc_descript.args, 0) # not doing offset indexing, so push 0
    else
      uliv[i] = findfirst(Base.Fix2(===,ind), looporder)::Int
      loop = getloop(ls, ind)
      if usegespsymbol
        if isknown(first(loop))
          push!(gespinds.args, Expr(:call, GlobalRef(Base, :(+)), gespsymbol, staticexpr(gethint(first(loop)))))
        else
          push!(gespinds.args, Expr(:call, GlobalRef(Base, :(+)), gespsymbol, getsym(first(loop))))
        end
      elseif isknown(first(loop))
        push!(gespinds.args, staticexpr(gethint(first(loop))))
      else
        push!(gespinds.args, getsym(first(loop)))
      end
      push!(offsetprecalc_descript.args, max(5,us.u₁,us.u₂))
      use_offsetprecalc = true
    end
  end
  arrayref_to_name_op = arrayref_to_name_op_collection[allarrayrefsind]
  if first(arrayref_to_name_op)[2] == 1 # if it's the first inside `array_refs_with_same_name`, we need to add the expression to the let block
    vptr_ar = if isone(length(li))
      # Workaround for fact that 1-d OffsetArrays are offset when using 1 index, but multi-dim ones are not
      Expr(:call, lv(:onetozeroindexgephack), vptrar)
    else
      vptrar
    end
    if use_offsetprecalc
      push!(q.args, Expr(:(=), vptrar, Expr(:call, lv(:offsetprecalc), Expr(:call, lv(:gesp), vptr_ar, gespinds), Expr(:call, Expr(:curly, :Val, offsetprecalc_descript)))))
    else
      push!(q.args, Expr(:(=), vptrar, Expr(:call, lv(:gesp), vptr_ar, gespinds)))
    end
  end
  uliv
end

# Plan here is that we increment every unique array
function add_loop_start_stop_manager!(ls::LoopSet)
    q = Expr(:block)
    us = ls.unrollspecification
    # Presence of an explicit use of a loopinducation var means we should use that, so we look for one
    # TODO: replace first with only once you add Compat as a dep or drop support for older Julia versions
    loopinductvars = Symbol[]
    for op ∈ operations(ls)
        isloopvalue(op) && push!(loopinductvars, first(loopdependencies(op)))
    end
    # Filtered ArrayReferenceMetas, we must increment each
    arrayrefs, name_to_array_map, unique_to_name_and_op_map = uniquearrayrefs(ls)
    use_livs = Vector{Vector{Int}}(undef, length(arrayrefs))
    # for i ∈ eachindex(name_to_array_map)
    for i ∈ eachindex(arrayrefs)
        use_livs[i] = use_loop_induct_var!(ls, q, arrayrefs[i], arrayrefs, i, name_to_array_map[first(first(unique_to_name_and_op_map[i]))], unique_to_name_and_op_map)
    end
    # @show use_livs,
    # loops, sorted from outer-most to inner-most
    looporder = reversenames(ls)
    # For each loop, we need to choose an induction variable
    nloops = length(looporder)
    # loopstarts = Vector{Vector{ArrayReferenceMeta}}(undef, nloops)
    loopstarts = fill(ArrayReferenceMeta[], nloops)
    terminators = Vector{Int}(undef, nloops) # zero is standard loop induct var
    # loopincrements = Vector{Vector{ArrayReferenceMeta}}(undef, nloops) # Not needed; copy of loopstarts
    # The purpose of the reminds thing here is to pick which of these to use for the terminator
    # We want to pick the one with the fewest outer loops with respect to this one, to minimize
    # the number of redefinitions of max-pointer used for the termination comparison.
    reached_indices = zeros(Int, length(arrayrefs))
    for (i,loopsym) ∈ enumerate(looporder) # iterates from outer to inner
        loopstartᵢ = ArrayReferenceMeta[]; arⱼ = 0; minrem = typemax(Int);
        ric = Tuple{Int,Int}[]
        for j ∈ eachindex(use_livs) # j is array ref number
            for (l,k) ∈ enumerate(use_livs[j])# l is index number, k is loop number
                if k == i
                    push!(loopstartᵢ, arrayrefs[j])
                    push!(ric, ((reached_indices[j] += 1), length(loopstartᵢ)))
                end
            end
        end
        loopstarts[nloops+1-i] = loopstartᵢ
        terminators[nloops+1-i] = if (loopsym ∈ loopinductvars) || (any(r -> any(isequal(-i), r), use_livs)) || iszero(length(loopstartᵢ))
            0
        else
            # @show i, loopsym loopdependencies.(operations(ls)) operations(ls)
            # @assert !iszero(length(loopstartᵢ))
            last(ric[argmin(first.(ric))]) # index corresponds to array ref's position in loopstart
        end
    end
    ls.lssm = LoopStartStopManager(
        terminators, loopstarts, arrayrefs
    )
    q
end
maxsym(ptr, sub) = Symbol(ptr, "##MAX##", sub, "##")
function pointermax(ls::LoopSet, ar::ArrayReferenceMeta, n::Int, sub::Int, isvectorized::Bool)::Expr
    pointermax(ls, ar, n, sub, isvectorized, getloop(ls, names(ls)[n]))
end
function pointermax(ls::LoopSet, ar::ArrayReferenceMeta, n::Int, sub::Int, isvectorized::Bool, loop::Loop)::Expr
    start = first(loop)
    stop = last(loop)
    incr = step(loop)
    if isknown(start) & isknown(stop)
        pointermax(ls, ar, n, sub, isvectorized, 1 + gethint(stop) - gethint(start), incr)
    end
    looplensym = isone(start) ? getsym(stop) : loop.lensym
    pointermax(ls, ar, n, sub, isvectorized, looplensym, incr)
end
function pointermax_index(
    ls::LoopSet, ar::ArrayReferenceMeta, n::Int, sub::Int, isvectorized::Bool, stophint::Int, incr::MaybeKnown
)::Tuple{Expr,Int}
    # @unpack u₁loopnum, u₂loopnum, vloopnum, u₁, u₂ = us
    loopsym = names(ls)[n]
    index = Expr(:tuple)
    found_loop_sym = false
    ind = 0
    for (j,i) ∈ enumerate(getindicesonly(ar))
        if i === loopsym
            ind = j
            if iszero(sub)
                push!(index.args, stophint)
            else
                _ind = if isvectorized
                    if isone(sub)
                        Expr(:call, lv(:vsub_fast), staticexpr(stophint), VECTORWIDTHSYMBOL)
                    else
                        Expr(:call, lv(:vsub_fast), staticexpr(stophint), mulexpr(VECTORWIDTHSYMBOL, sub))
                    end
                else
                    staticexpr(stophint - sub)
                end
                stride = getstrides(ar)[j]
                if isknown(incr)
                    stride *= gethint(incr)
                else
                    _ind = mulexpr(_ind, getsym(incr))
                end
                if stride ≠ 1
                    @assert stride ≠ 0 "stride shouldn't be 0 if used for determining loop start/stop, but loop $n array $ar was."
                    _ind = lazymulexpr(stride, _ind)
                end
                push!(index.args, _ind)
            end
        else
            push!(index.args, Expr(:call, lv(:Zero)))
        end
    end
    @assert ind ≠ 0 "Failed to find $loopsym"
    index, ind
end
function pointermax_index(ls::LoopSet, ar::ArrayReferenceMeta, n::Int, sub::Int, isvectorized::Bool, stopsym, incr::MaybeKnown)::Tuple{Expr,Int}
    loopsym = names(ls)[n]
    index = Expr(:tuple);
    ind = 0
    # @show ar loopsym names(ls) n
    for (j,i) ∈ enumerate(getindicesonly(ar))
        # @show j,i
        if i === loopsym
            ind = j
            if iszero(sub)
                push!(index.args, stopsym)
            else
                _ind = if isvectorized
                    if isone(sub)
                        Expr(:call, lv(:vsub_fast), stopsym, VECTORWIDTHSYMBOL)
                    else
                        Expr(:call, lv(:vsub_fast), stopsym, mulexpr(VECTORWIDTHSYMBOL, sub))
                    end
                else
                     subexpr(stopsym, sub)
                end
                stride = getstrides(ar)[j]
                if isknown(incr)
                    stride *= gethint(incr)
                else
                    _ind = mulexpr(_ind, getsym(incr))
                end
                if stride ≠ 1
                    @assert stride ≠ 0 "stride shouldn't be 0 if used for determining loop start/stop, but loop $n array $ar was."
                    _ind = lazymulexpr(stride, _ind)
                end
                push!(index.args, _ind)
            end
        else
            push!(index.args, Expr(:call, lv(:Zero)))
        end
    end
    @assert ind != 0 "Failed to find $loopsym"
    index, ind
end
function pointermax(ls::LoopSet, ar::ArrayReferenceMeta, n::Int, sub::Int, isvectorized::Bool, stopsym, incr::MaybeKnown)::Expr
    index = first(pointermax_index(ls, ar, n, sub, isvectorized, stopsym, incr))
    Expr(:call, lv(:gesp), vptr(ar), index)
end

function defpointermax(ls::LoopSet, ar::ArrayReferenceMeta, n::Int, sub::Int, isvectorized::Bool)::Expr
    Expr(:(=), maxsym(vptr(ar), sub), pointermax(ls, ar, n, sub, isvectorized))
end
function offsetindex(dim::Int, ind::Int, scale::Int, isvectorized::Bool, incr::MaybeKnown)
    index = Expr(:tuple)
    for d ∈ 1:dim
        if d ≠ ind || iszero(scale)
            push!(index.args, Expr(:call, lv(:Zero)))
            continue
        end
        if !isvectorized
            pushmulexpr!(index, scale, incr)
        elseif isone(scale)
            pushmulexpr!(index, VECTORWIDTHSYMBOL, incr)
        else
            push!(index.args, mulexpr(VECTORWIDTHSYMBOL, scale, incr))
        end
    end
    index
end
function append_pointer_maxes!(
    loopstart::Expr, ls::LoopSet, ar::ArrayReferenceMeta, n::Int, submax::Int, isvectorized::Bool, stopindicator, incr::MaybeKnown
)
    vptr_ar = vptr(ar)
    if submax < 2
        for sub ∈ 0:submax
            push!(loopstart.args, Expr(:(=), maxsym(vptr_ar, sub), pointermax(ls, ar, n, sub, isvectorized, stopindicator, incr)))
        end
    else
        # @show n, getloop(ls, n) ar
        index, ind = pointermax_index(ls, ar, n, submax, isvectorized, stopindicator, incr)
        pointercompbase = maxsym(vptr_ar, submax)
        push!(loopstart.args, Expr(:(=), pointercompbase, Expr(:call, lv(:gesp), vptr_ar, index)))
        dim = length(getindicesonly(ar))
        # OFFSETPRECALCDEF = true
        # if OFFSETPRECALCDEF
        strd = getstrides(ar)[ind]
        for sub ∈ 0:submax-1
            ptrcmp = Expr(:call, lv(:gesp), pointercompbase, offsetindex(dim, ind, (submax - sub)*strd, isvectorized, incr))
            push!(loopstart.args, Expr(:(=), maxsym(vptr_ar, sub), ptrcmp))
        end
        # else
        #     indexoff = offsetindex(dim, ind, 1, isvectorized)
        #     for sub ∈ submax-1:-1:0
        #         _newpointercompbase = maxsym(vptr_ar, sub)
        #         newpointercompbase = gensym(_pointercompbase)
        #         push!(loopstart.args, Expr(:(=), newpointercompbase, Expr(:call, lv(:gesp), pointercompbase, indexoff)))
        #         push!(loopstart.args, Expr(:(=), _newpointercompbase, Expr(:call, lv(:pointerforcomparison), newpointercompbase)))
        #         _pointercompbase = _newpointercompbase
        #         pointercompbase = newpointercompbase
        #     end
        # end
    end
end
function append_pointer_maxes!(loopstart::Expr, ls::LoopSet, ar::ArrayReferenceMeta, n::Int, submax::Int, isvectorized::Bool)
    loop = getloop(ls, n)
    @assert loop.itersymbol == names(ls)[n]
    start = first(loop)
    stop = last(loop)
    incr = step(loop)
    if isknown(start) & isknown(stop)
        return append_pointer_maxes!(loopstart, ls, ar, n, submax, isvectorized, startstopΔ(loop)+1, incr)
    end
    looplensym = isone(start) ? getsym(stop) : loop.lensym
    append_pointer_maxes!(loopstart, ls, ar, n, submax, isvectorized, looplensym, incr)
end

function maxunroll(us::UnrollSpecification, n)
    @unpack u₁loopnum, u₂loopnum, u₁, u₂ = us
    if n == u₁loopnum# && u₁ > 1
        u₁
    elseif n == u₂loopnum# && u₂ > 1
        u₂
    else
        1
    end
end


function startloop(ls::LoopSet, us::UnrollSpecification, n::Int, submax = maxunroll(us, n))
    @unpack u₁loopnum, u₂loopnum, vloopnum, u₁, u₂ = us
    lssm = ls.lssm
    termind = lssm.terminators[n]
    ptrdefs = lssm.incrementedptrs[n]
    loopstart = Expr(:block)
    firstloop = n == num_loops(ls)
    for ar ∈ ptrdefs
        ptr = vptr(ar)
        push!(loopstart.args, Expr(:(=), ptr, ptr))
    end
    if iszero(termind)
        loopsym = names(ls)[n]
        push!(loopstart.args, startloop(getloop(ls, loopsym), loopsym))
    else
        isvectorized = n == vloopnum
        # @show ptrdefs
        append_pointer_maxes!(loopstart, ls, ptrdefs[termind], n, submax, isvectorized)
    end
    loopstart
end
function offset_ptr(
    ar::ArrayReferenceMeta, us::UnrollSpecification, loopsym::Symbol, n::Int, UF::Int, offsetinds::Vector{Bool}, loop::Loop
)
    indices = getindices(ar)
    strides = getstrides(ar)
    offset = first(indices) === DISCONTIGUOUS
    gespinds = Expr(:tuple)
    li = ar.loopedindex
    for i ∈ eachindex(li)
        ii = i + offset
        ind = indices[ii]
        if !offsetinds[i] || ind !== loopsym
            push!(gespinds.args, Expr(:call, lv(:Zero)))
        else
            incrementloopcounter!(gespinds, us, n, UF * strides[i], loop)
        end
        # ind == loopsym && break
    end
    Expr(:(=), vptr(ar), Expr(:call, lv(:gesp), vptr(ar), gespinds))
end
function incrementloopcounter!(q::Expr, ls::LoopSet, us::UnrollSpecification, n::Int, UF::Int)
    @unpack u₁loopnum, u₂loopnum, vloopnum, u₁, u₂ = us
    lssm = ls.lssm
    ptrdefs = lssm.incrementedptrs[n]
    looporder = names(ls)
    loopsym = looporder[n]
    termind = lssm.terminators[n]
    loop = getloop(ls, n)
    if iszero(termind) # increment liv
        push!(q.args, incrementloopcounter(us, n, loopsym, UF, loop))
    end
    for (j,ar) ∈ enumerate(ptrdefs)
        offsetinds = indices_calculated_by_pointer_offsets(ls, ar)
        push!(q.args, offset_ptr(ar, us, loopsym, n, UF, offsetinds, loop))
    end
    nothing
end
function terminatecondition(ls::LoopSet, us::UnrollSpecification, n::Int, inclmask::Bool, UF::Int)
    lssm = ls.lssm
    termind = lssm.terminators[n]
    if iszero(termind)
        loop = getloop(ls, n)
        return terminatecondition(loop, us, n, loop.itersymbol, inclmask, UF)
    end

    termar = lssm.incrementedptrs[n][termind]
    ptr = vptr(termar)
    # @show UF, isvectorized(us, n)
    if inclmask && isvectorized(us, n)
        Expr(:call, :<, ptr, maxsym(ptr, 0))
    else
        Expr(:call, :≤, ptr, maxsym(ptr, UF))
    end
end
