


function uniquearrayrefs_csesummary(ls::LoopSet)
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

function uniquearrayrefs(ls::LoopSet)
    uniquerefs = ArrayReferenceMeta[]
    includeinlet = Bool[]
    # for arrayref ∈ ls.refs_aliasing_syms
    for op ∈ operations(ls)
        arrayref = op.ref
        arrayref === NOTAREFERENCE && continue
        notunique = false
        isonlyname = true
        for ref ∈ uniquerefs
            notunique = sameref(arrayref, ref)
            isonlyname &= vptr(arrayref) !== vptr(ref)
            # if they're not the sameref, they may still have the same name
            # if they have different names, they're definitely not sameref
            notunique && break
        end
        if !notunique
            push!(uniquerefs, arrayref)
            push!(includeinlet, isonlyname)
        end
    end
    uniquerefs, includeinlet
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

# @generated function set_first_stride(sptr::StridedPointer{T,N,C,B,R}) where {T,N,C,B,R}
#     minrank = argmin(R)
#     newC = C > 0 ? (C == minrank ? 1 : 0) : C
#     newB = C > 0 ? (C == minrank ? B : 0) : B #TODO: confirm correctness
#     quote
#         $(Expr(:meta,:inline))
#         # VectorizationBase.StridedPointer{$T,1,$newC,$newB,$(R[minrank],)}($(lv(llvmptr))(sptr), (sptr.strd[$minrank],), (Zero(),))
#         VectorizationBase.StridedPointer{$T,1,$newC,$newB,$(R[minrank],)}(VectorizationBase.cpupointer(sptr), (sptr.strd[$minrank],), (Zero(),))
#     end
# end
# set_first_stride(x) = x # cross fingers that this works
# @inline onetozeroindexgephack(sptr::AbstractStridedPointer) = gesp(set_first_stride(sptr), (Static{-1}(),)) # go backwords
# @inline onetozeroindexgephack(sptr::AbstractStridedPointer{T,1}) where {T} = sptr
# @inline onetozeroindexgephack(sptr::StridedPointer{T,1}) where {T} = sptr
# @inline onetozeroindexgephack(x) = x

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
function normalize_offsets!(
  ls::LoopSet, i::Int, allarrayrefs::Vector{ArrayReferenceMeta},
  array_refs_with_same_name::Vector{Int}, arrayref_to_name_op_collection::Vector{Vector{Tuple{Int,Int,Int}}}
)
  ops = operations(ls)
  length(ops) > 128 && return 0
  minoffset::Int8 = typemax(Int8)
  maxoffset::Int8 = typemin(Int8)
  # we want to store the offsets, because we don't want to require that the `offset` vectors of the variaous `ArrayReferenceMeta`s don't alias
  offsets::Base.RefValue{NTuple{128,Int8}} = Base.RefValue{NTuple{128,Int8}}();
  GC.@preserve offsets begin
    poffsets = Base.unsafe_convert(Ptr{Int8}, offsets)
    for j ∈ array_refs_with_same_name
      arrayref_to_name_op = arrayref_to_name_op_collection[j]
      for (_,__,opid) ∈ arrayref_to_name_op
        op = ops[opid]
        off = getoffsets(op.ref)[i]
        off == zero(Int8) && return 0
        minoffset = min(off, minoffset)
        maxoffset = max(off, maxoffset)
        unsafe_store!(poffsets, off, opid)
      end
    end
    # reaching here means none of the offsets contain `0`
    # we won't bother if difference between offsets is >127
    # we don't want `maxoffset` to overflow when subtracting `minoffset`
    # so we check if it's safe, and give up if it isn't
    (Int(maxoffset) - Int(minoffset)) > 127 && return 0
    for j ∈ array_refs_with_same_name
      arrayref_to_name_op = arrayref_to_name_op_collection[j]
      for (_,__,opid) ∈ arrayref_to_name_op
        getoffsets(ops[opid].ref)[i] = unsafe_load(poffsets, opid) - minoffset
      end
    end
  end
  return Int(minoffset)
end
function isloopvalue(ls::LoopSet, ind::Symbol, isrooted::Union{Nothing,Vector{Bool}} = nothing)
  for (i,op) ∈ enumerate(operations(ls))
    if (isrooted ≢ nothing)
      isrooted[i] || continue
    end
    iscompute(op) || continue
    for opp ∈ parents(op)# this is to confirm `ind` still has children
      # (isloopvalue(opp) && instruction(opp).instr === ind) && return true
      if (isloopvalue(opp) && instruction(opp).instr === ind)
        return true
      end
    end
  end
  return false
end
function cse_constant_offsets!(
  ls::LoopSet, allarrayrefs::Vector{ArrayReferenceMeta}, allarrayrefsind::Int, name_to_array_map::Vector{Vector{Int}}, arrayref_to_name_op_collection::Vector{Vector{Tuple{Int,Int,Int}}}
)
  ar = allarrayrefs[allarrayrefsind]
  # vptrar = vptr(ar)
  arrayref_to_name_op = arrayref_to_name_op_collection[allarrayrefsind]
  array_refs_with_same_name = name_to_array_map[first(first(arrayref_to_name_op))]
  us = ls.unrollspecification
  li = ar.loopedindex
  indices = getindices(ar)
  strides = getstrides(ar)
  offset = first(indices) === DISCONTIGUOUS
  # gespindoffsets = fill(Symbol(""), length(li))
  gespindsummary = Vector{Tuple{Symbol,Int}}(undef, length(li))
  for i ∈ eachindex(li)
    gespsymbol::Symbol = Symbol("")    
    ii = i + offset
    ind::Symbol = indices[ii]
    licmoffset = !li[i]
    if licmoffset
      # we try to licm and offset so we can set `li[i] = true`
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
      ops = operations(ls)
      while licmoffset # repeat until we run out
        # ind = indices[ii]
        # indices are all the same across operations, so we look to the first for checking compatibility...
        memop = indop = ops[first(arrayref_to_name_op)[3]] # indop is a dummy placeholder
        for opp ∈ parents(memop)
          if name(opp) === ind
            indop = opp
          end
        end
        # (we found a match) # iscompute(indop) should be guaranteed...
        ((indop ≢ memop) && iscompute(indop)) || break
        instr = instruction(indop).instr
        parents_indop = parents(indop)
        if instr === :(+)
          constants_to_licm = if gespsymbol === Symbol("")
            Expr(:call, GlobalRef(Base,:(+)))
          else
            Expr(:call, GlobalRef(Base,:(+)), gespsymbol)
          end
          new_parent = indop # dummy, for now we will only licm if it lets us remove `indop`
          for opp ∈ parents_indop
            if isconstantop(opp)
              # NOTE: using `name` because this happens at macroexpansion time
              # push!(constants_to_licm.args, constantopname(opp))
              push!(constants_to_licm.args, name(opp))
            elseif new_parent === indop # first
              new_parent = opp
            else # this is the second parent, we give up on `licm`
              licmoffset = false
              break
            end
          end
          licmoffset || continue
          if length(constants_to_licm.args) > 2
            gespsymbol = gensym!(ls, "#gespsym#")
            pushpreamble!(ls, Expr(:(=), gespsymbol, constants_to_licm))
          elseif length(constants_to_licm.args) == 2
            licmoffset = gespsymbol === Symbol("")
            if licmoffset
              gespsymbol = (constants_to_licm.args[2])::Symbol
            end
          else
            licmoffset = false
          end
          if licmoffset # prune
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
              subexpr = if gespsymbol === Symbol("")
                Expr(:call, GlobalRef(Base,:(-)), name(op1), name(op2))
              else
                Expr(:call, GlobalRef(Base,:(-)), Expr(:call, GlobalRef(Base,:(+)), gespsymbol, name(op1)), name(op2))
              end
              gespsymbol = gensym!(ls, "#gespsym#")
              pushpreamble!(ls, Expr(:(=), gespsymbol, subexpr))
              ind = CONSTANTZEROINDEX
              set_all_to_constant_index!(ls, i, ii, indop, allarrayrefs, array_refs_with_same_name, arrayref_to_name_op_collection)
            else# op1const, op2dynamic
              # won't bother with this for now
              licmoffset = false
            end
          elseif op2const #op1dynamic
            subexpr = if gespsymbol === Symbol("")
              Expr(:call, GlobalRef(Base,:(-)), name(op2))
            else
              Expr(:call, GlobalRef(Base,:(-)), gespsymbol, name(op2))
            end
            gespsymbol = gensym!(ls, "#gespsym#")
            pushpreamble!(ls, Expr(:(=), gespsymbol, subexpr))
            ind = maybeloopvaluename(op1)
            substitute_ops_all!(ls, i, ii, indop, op1, allarrayrefs, array_refs_with_same_name, arrayref_to_name_op_collection)
          else
            licmoffset = false
          end
        else
          licmoffset = false
        end
      end
    end
    constoffset = normalize_offsets!(ls, i, allarrayrefs, array_refs_with_same_name, arrayref_to_name_op_collection)
    gespindsummary[i] = (gespsymbol, constoffset)
    # pushgespind!(gespinds, ls, gespsymbol, constoffset, ind, li, i, check_shouldindbyind(ls, ind, shouldindbyind), true)
  end
  return gespindsummary
end
@inline similardims(_, i) = i
@inline similardims(::CartesianIndices{N}, i) where {N} = VectorizationBase.CartesianVIndex(ntuple(_ -> i, Val{N}()))
# @generated function similardims(::CartesianIndices{N}, i) where {N}
#   t = Expr(:tuple)
#   for n ∈ 1:N
#     push!(t.args, :i)
#   end
#   Expr(:block,Expr(:meta,:inline),t)
# end
# function maybebroadcastpush!(q::Expr, loop::Loop, b::Bool, ex)
#   if b
#     push!(q.args, Expr(:call, lv(:similardims), loop.rangesym, ex))
#   else
#     push!(q.args, ex)
#   end
#   return nothing
# end
function calcgespinds(ls::LoopSet, ar::ArrayReferenceMeta, gespindsummary::Vector{Tuple{Symbol,Int}}, shouldindbyind::Vector{Bool})
  gespinds = Expr(:tuple)
  li = ar.loopedindex
  indices = getindicesonly(ar)
  for i ∈ eachindex(li)
    ind = indices[i]
    gespsymbol, constoffset = gespindsummary[i]
    pushgespind!(gespinds, ls, gespsymbol, constoffset, ind, li[i], check_shouldindbyind(ls, ind, shouldindbyind), true)
  end
  gespinds
end

function pushgespind!(
  gespinds::Expr, ls::LoopSet, gespsymbol::Symbol, constoffset::Int, ind::Symbol, isli::Bool, index_by_index::Bool, fromgsp::Bool
)
  if isli
    if ind === CONSTANTZEROINDEX
      if gespsymbol === Symbol("")
        push!(gespinds.args, staticexpr(constoffset))
      elseif constoffset == 0
        push!(gespinds.args, gespsymbol)
      else
        push!(gespinds.args, Expr(:call, GlobalRef(Base,:(+)), gespsymbol, staticexpr(constoffset)))
      end
    else
      if index_by_index
        if gespsymbol === Symbol("")
          if constoffset == 0
            ns = Expr(:call, GlobalRef(VectorizationBase, :NullStep))
            if fromgsp
              loop = getloop(ls, ind)
              if loop.rangesym ≢ Symbol("")
                ns = Expr(:call, lv(:similardims), loop.rangesym, ns)
              end
            end
            push!(gespinds.args, ns)
          else
            push!(gespinds.args, staticexpr(constoffset))
          end
        elseif constoffset == 0
          push!(gespinds.args, gespsymbol)
        else
          push!(gespinds.args, addexpr(gespsymbol, constoffset))
        end
      else
        loop = getloop(ls, ind)
        if gespsymbol === Symbol("")
          if isknown(first(loop))
            push!(gespinds.args, staticexpr(constoffset + gethint(first(loop))))
          elseif constoffset == 0
            push!(gespinds.args, getsym(first(loop)))
          else
            push!(gespinds.args, addexpr(getsym(first(loop)), constoffset))
          end
        elseif isknown(first(loop))
          loopfirst = gethint(first(loop)) + constoffset
          if loopfirst == 0
            push!(gespinds.args, gespsymbol)
          else
            push!(gespinds.args, Expr(:call, GlobalRef(Base, :(+)), gespsymbol, staticexpr(loopfirst)))
          end
        else
          addedstarts = Expr(:call, GlobalRef(Base, :(+)), gespsymbol, getsym(first(loop)))
          if constoffset == 0
            push!(gespinds.args, addedstarts)
          else
            push!(gespinds.args, Expr(:call, GlobalRef(Base, :(+)), addedstarts, staticexpr(constoffset)))
          end
        end
      end
    end # if we hit the following elseif/else, not a loopindex
  elseif fromgsp # from gsp means that a loop could be a CartesianIndices, so we may need to expand
    #TODO: broadcast dimensions in case of cartesian indices
    rangesym = ind
    foundind = false
    for op ∈ operations(ls)
      if name(op) === ind
        loopdeps = loopdependencies(op)
        foundind = true
        if length(loopdeps) ≠ 0
          rangesym = getloop(ls, first(loopdeps)).rangesym
        else
          isconstantop(op) || throw(LoopError("Please file an issue with LoopVectorization.jl with a reproducer; tried to eliminate a non-constant operation."))
          rangesym = name(op)
        end
        break
      end
    end
    @assert foundind
    if rangesym === Symbol("") # there is no rangesym, must be statically sized.
      pushgespsym!(gespinds, gespsymbol, constoffset)
    else
      pushsimdims!(gespinds, rangesym, gespsymbol, constoffset)
    end
  else # it is known all inds are 1-dimensional
    pushgespsym!(gespinds, gespsymbol, constoffset)
  end
  return nothing
end
function pushgespsym!(gespinds::Expr, gespsymbol::Symbol, constoffset::Int)
  if gespsymbol === Symbol("")
    push!(gespinds.args, staticexpr(constoffset))
  elseif constoffset == 0
    push!(gespinds.args, gespsymbol)
  else
    push!(gespinds.args, addexpr(gespsymbol, constoffset))
  end
  return nothing
end
function pushsimdims!(gespinds::Expr, rangesym::Symbol, gespsymbol::Symbol, constoffset::Int)
  simdimscall = Expr(:call, lv(:similardims), rangesym)
  pushgespsym!(simdimscall, gespsymbol, constoffset)
  push!(gespinds.args, simdimscall)
  return nothing
end

# if using ptr offsets, need to offset by loopstart
# if calculating by index, do not offset by loopstart
"""
Returns a vector of length equal to the number of indices.
A value > 0 indicates which loop number that index corresponds to when incrementing the pointer.
A value < 0 indicates that abs(value) is the corresponding loop, and a `loopvalue` will be used.
"""
function use_loop_induct_var!(
  ls::LoopSet, q::Expr, ar::ArrayReferenceMeta, allarrayrefs::Vector{ArrayReferenceMeta}, allarrayrefsind::Int, includeinlet::Bool
  # array_refs_with_same_name::Vector{Int}, arrayref_to_name_op_collection::Vector{Vector{Tuple{Int,Int,Int}}}
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
  # if (!isbroadcast) && !(all(li))
  #   checkforgespind = true
  #   gespindoffsets = cse_constant_offsets!(ls, q, ar, allarrayrefs, allarrayrefsind, array_refs_with_same_name, arrayref_to_name_op_collection)
  # else
  #   checkforgespind = false
  #   gespindoffsets = indices#dummy
  # end
  gespinds = Expr(:tuple)
  offsetprecalc_descript = Expr(:tuple)
  use_offsetprecalc = false
  vptrar = vptr(ar)
  Wisz = false#ls.vector_width == 0
  for (i,isli) ∈ enumerate(li)
    ii = i + offset
    ind = indices[ii]
    Wisz && push!(gespinds.args, staticexpr(0)) # wrong for `@_turbo`...
    if !li[i] # if it wasn't set
      uliv[i] = 0
      push!(offsetprecalc_descript.args, 0)
      Wisz || pushgespind!(gespinds, ls, Symbol(""), 0, ind, isli, true, false)
    elseif ind === CONSTANTZEROINDEX
      uliv[i] = 0
      push!(offsetprecalc_descript.args, 0)
      Wisz || pushgespind!(gespinds, ls, Symbol(""), 0, ind, isli, true, false)
    elseif isbroadcast ||
      ((isone(ii) && (last(looporder) === ind)) && !(otherindexunrolled(ls, ind, ar)) ||
      multiple_with_name(vptrar, allarrayrefs)) ||
      (iszero(ls.vector_width) && isstaticloop(getloop(ls, ind)))# ||
      # Not doing normal offset indexing
      uliv[i] = -findfirst(Base.Fix2(===,ind), looporder)::Int
      push!(offsetprecalc_descript.args, 0) # not doing offset indexing, so push 0
      Wisz || pushgespind!(gespinds, ls, Symbol(""), 0, ind, isli, true, false)
    else
      uliv[i] = findfirst(Base.Fix2(===,ind), looporder)::Int
      loop = getloop(ls, ind)
      push!(offsetprecalc_descript.args, max(5,us.u₁+1,us.u₂+1))
      use_offsetprecalc = true
      Wisz || pushgespind!(gespinds, ls, Symbol(""), 0, ind, isli, false, false)
    end
    # cases for pushgespind! and loopval!
    # if !isloopval, same as before
    # if isloopval & `uliv[i] > 0`, then must setup
  end
  # arrayref_to_name_op = arrayref_to_name_op_collection[allarrayrefsind]
  if includeinlet#first(arrayref_to_name_op)[2] == 1 # if it's the first inside `array_refs_with_same_name`, we need to add the expression to the let block
    vpgesped = Expr(:call, lv(:gesp), vptrar, gespinds)
    if use_offsetprecalc
      vpgesped = Expr(:call, lv(:offsetprecalc), vpgesped, Expr(:call, Expr(:curly, :Val, offsetprecalc_descript)))
    end
    push!(q.args, Expr(:(=), vptrar, vpgesped))
    push!(q.args, Expr(:(=), vptr_offset(vptrar), Expr(:call, GlobalRef(VectorizationBase, :increment_ptr), vptrar)))
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
  # arrayrefs, name_to_array_map, unique_to_name_and_op_map = uniquearrayrefs(ls)
    arrayrefs, includeinlet = uniquearrayrefs(ls)
    use_livs = Vector{Vector{Int}}(undef, length(arrayrefs))
    # for i ∈ eachindex(name_to_array_map)
    for i ∈ eachindex(arrayrefs)
      use_livs[i] = use_loop_induct_var!(ls, q, arrayrefs[i], arrayrefs, i, includeinlet[i])
      #name_to_array_map[first(first(unique_to_name_and_op_map[i]))], unique_to_name_and_op_map)
    end
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
        return pointermax(ls, ar, n, sub, isvectorized, 1 + gethint(stop) - gethint(start), incr)
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
                        Expr(:call, lv(:vsub_nsw), staticexpr(stophint), VECTORWIDTHSYMBOL)
                    else
                        Expr(:call, lv(:vsub_nsw), staticexpr(stophint), mulexpr(VECTORWIDTHSYMBOL, sub))
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
    for (j,i) ∈ enumerate(getindicesonly(ar))
        if i === loopsym
            ind = j
            if iszero(sub)
                push!(index.args, stopsym)
            else
                _ind = if isvectorized
                    if isone(sub)
                        Expr(:call, lv(:vsub_nsw), stopsym, VECTORWIDTHSYMBOL)
                    else
                        Expr(:call, lv(:vsub_nsw), stopsym, mulexpr(VECTORWIDTHSYMBOL, sub))
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
  vptrar = vptr(ar)
  Expr(:call, GlobalRef(VectorizationBase,:increment_ptr), vptrar, vptr_offset(vptrar), index)
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
    index, ind = pointermax_index(ls, ar, n, submax, isvectorized, stopindicator, incr)
    pointercompbase = maxsym(vptr_ar, submax)
    ip = GlobalRef(VectorizationBase, :increment_ptr)  
    push!(loopstart.args, Expr(:(=), pointercompbase, Expr(:call, ip, vptr_ar, vptr_offset(vptr_ar), index)))
    dim = length(getindicesonly(ar))
    # OFFSETPRECALCDEF = true
    # if OFFSETPRECALCDEF
    strd = getstrides(ar)[ind]
    for sub ∈ 0:submax-1
      ptrcmp = Expr(:call, ip, vptr_ar, pointercompbase, offsetindex(dim, ind, (submax - sub)*strd, isvectorized, incr))
      push!(loopstart.args, Expr(:(=), maxsym(vptr_ar, sub), ptrcmp))
    end
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


function startloop(ls::LoopSet, us::UnrollSpecification, n::Int, staticinit::Bool = false)
  @unpack u₁loopnum, u₂loopnum, vloopnum, u₁, u₂ = us
  lssm = ls.lssm
  termind = lssm.terminators[n]
  ptrdefs = lssm.incrementedptrs[n]
  loopstart = Expr(:block)
  firstloop = n == num_loops(ls)
  for ar ∈ ptrdefs
    ptr_offset = vptr_offset(ar)
    push!(loopstart.args, Expr(:(=), ptr_offset, ptr_offset))
  end
  if iszero(termind)
    loopsym = names(ls)[n]
    push!(loopstart.args, startloop(getloop(ls, loopsym), loopsym, staticinit))
  else
    isvectorized = n == vloopnum
    submax = maxunroll(us, n)
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
  vpoff = vptr_offset(ar)
  call = Expr(:call, GlobalRef(VectorizationBase, :increment_ptr), vptr(ar), vpoff, gespinds)
  Expr(:(=), vpoff, call)
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
  optr = vptr_offset(ptr)
  if inclmask && isvectorized(us, n)
    Expr(:call, GlobalRef(VectorizationBase, :vlt), optr, maxsym(ptr, 0), ptr)
  else
    Expr(:call, GlobalRef(VectorizationBase, :vle), optr, maxsym(ptr, UF), ptr)
  end
end
