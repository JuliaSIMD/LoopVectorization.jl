function prefetchisagoodidea(ls::LoopSet, op::Operation, td::UnrollArgs)
    # ((Sys.ARCH === :x86_64) || (Sys.ARCH === :i686)) || return false
    # return false
    @unpack u₁, u₁loopsym, u₂loopsym, vloopsym, vstep, u₂max, suffix = td
    length(loopdependencies(op)) ≤ 1 && return 0
    isvectorized(op) || return 0
    ((u₁ > 1) & (u₂max > 1)) || return 0
    u₂loopsym === Symbol("##undefined##") && return 0
    # @show cache_lnsze(ls) reg_size(ls) pointer_from_objref(ls.register_size)
    dontskip = (cache_lnsze(ls) ÷ reg_size(ls)) - 1
    # u₂loopsym is vectorized
    # u₁vectorized = vectorized === u₁loopsym
    u₂vectorized = vloopsym === u₂loopsym
    ((suffix ≠ -1) && u₂vectorized && !iszero(suffix & dontskip)) && return 0
    innermostloopsym = first(names(ls))
    loopedindex = op.ref.loopedindex
    if length(loopedindex) > 1 && first(loopedindex)
        indices = getindices(op)
        if first(indices) === vloopsym && first(indices) !== innermostloopsym
            (isone(vstep) && isone(first(getstrides(op)))) || return 0
            # We want at least 4 reuses per load
            uses = Core.ifelse(isu₁unrolled(op), 1, u₁)
            uses = Core.ifelse(isu₂unrolled(op), uses, uses * u₂max)
            uses < 4 && return 0
            innermostloopind = -1
            for (i,inds) ∈ enumerate(getindices(op))
                if inds === innermostloopsym
                    innermostloopind == -1 || return 0
                    innermostloopind = i
                end
            end
            innermostloopind == -1 && return 0
          # if prod(s -> Float64(looplength(ls, s)), @view(indices[1:innermostloopind-1])) ≥ 120.0 &&
            if length(getloop(ls, innermostloopsym)) ≥ 120
                if getoffsets(op)[innermostloopind] < 120
                    for opp ∈ operations(ls)
                        if iscompute(opp) && (innermostloopsym ∈ loopdependencies(opp)) &&
                            load_constrained(opp, u₁loopsym, u₂loopsym, innermostloopsym, true)
                            return 0
                        end
                    end
                    return innermostloopind
                end
            end
        end
    end
    0
end
function add_prefetches!(q::Expr, ls::LoopSet, op::Operation, td::UnrollArgs, prefetchind::Int)
    # TODO: maybe prefetch for non-x86_64?
    ((Sys.ARCH === :x86_64) || (Sys.ARCH === :i686)) || return nothing
    @unpack u₁, u₁loopsym, u₂loopsym, vloopsym, u₂max = td
    # we should only be here if `unitsride(vloop)`
    dontskip = (cache_lnsze(ls) ÷ reg_size(ls)) - 1
    ptr = vptr(op)
    innermostloopsym = first(names(ls))
    us = ls.unrollspecification
    prefetch_distance = u₁loopsym === innermostloopsym ? us.u₁ : ( u₂loopsym === innermostloopsym ? us.u₂ : 1 )
    # prefetch_distance = u₁loopsym === innermostloopsym ? u₁ : ( u₂loopsym === innermostloopsym ? u₂max : 1 )
    prefetch_multiplier = 5
    prefetch_distance *= prefetch_multiplier
    offsets = getoffsets(op) # what a hack
    inner_offset = offsets[prefetchind]
    prefetchstride = prefetch_distance * getstrides(op)[prefetchind]
    prefetchloop_step = step(getloop(ls, getindices(op)[prefetchind]))
    if isknown(prefetchloop_step)
        prefetchstride *= gethint(prefetchloop_step)
    end
    offsets[prefetchind] = inner_offset + prefetchstride
    gespinds = mem_offset_u(op, td, indices_calculated_by_pointer_offsets(ls, op.ref), false, 0, ls)
    offsets[prefetchind] = inner_offset
    ptr = vptr(op)
    gptr = Symbol(ptr, "##GESPEDPREFETCH##")
    if !isknown(prefetchloop_step)
        for i ∈ eachindex(gespinds.args)
            if i == prefetchind
                gespinds.args[i] = mulexpr(getsym(prefetchloop_step), gespinds.args[i])
            end
            # gespinds.args[i] = Expr(:call, lv(:data), gespinds.args[i])
        end
    end
    ip = GlobalRef(VectorizationBase, :increment_ptr)
    push!(q.args, Expr(:(=), gptr, Expr(:call, ip, ptr, vptr_offset(ptr), gespinds)))
    inds = Expr(:tuple)
    indices = getindicesonly(op)

    i = 0
    for (j,ind) ∈ enumerate(indices)
        push!(inds.args, Expr(:call, lv(:Zero)))
        (ind == u₁loopsym) && (i = j)
    end
    prefetch0 = GlobalRef(VectorizationBase, :prefetch)
    push!(q.args, Expr(:call, prefetch0, Expr(:call, ip, ptr, gptr, copy(inds))))
    # push!(q.args, Expr(:call, lv(:prefetch0), gptr, copy(inds)))
    i == 0 && return
    for u ∈ 1:u₁-1
        # for u ∈ umin:min(umin,U-1)
        # (u₁loopsym === vloopsym && !iszero(u & dontskip)) && continue
        if u₁loopsym === vloopsym
            iszero(u & dontskip) || continue
            # W = ls.vector_width[]
            # if W != 0
            #     inds.args[i] = staticexpr(W*u)
            # else
            if isone(u)
                inds.args[i] = VECTORWIDTHSYMBOL
            else
                inds.args[i] = mulexpr(VECTORWIDTHSYMBOL, u)
            end
        else
            inds.args[i] = staticexpr(u)
        end
        push!(q.args, Expr(:call, prefetch0, Expr(:call, ip, ptr, gptr, copy(inds))))
    end
    nothing
end
broadcastedname(mvar) = Symbol(mvar, "##broadcasted##")
function pushbroadcast!(q::Expr, mvar::Symbol)
    push!(q.args, Expr(:(=), broadcastedname(mvar), Expr(:call, lv(:vbroadcast), VECTORWIDTHSYMBOL, mvar)))
end

function lower_load_no_optranslation!(
    q::Expr, ls::LoopSet, op::Operation, td::UnrollArgs, mask::Bool, inds_calc_by_ptr_offset::Vector{Bool}
)
    @unpack u₁, u₁loopsym, u₂loopsym, vloopsym, suffix = td
    loopdeps = loopdependencies(op)
    # @assert isvectorized(op)
    opu₁, opu₂ = isunrolled_sym(op, u₁loopsym, u₂loopsym, vloopsym, ls)
    u = ifelse(opu₁, u₁, 1)
    mvar = Symbol(variable_name(op, Core.ifelse(opu₂, suffix,-1)), '_', u)
    falseexpr = Expr(:call, lv(:False)); rs = staticexpr(reg_size(ls))
    if all(op.ref.loopedindex) && !rejectcurly(op)
        inds = unrolledindex(op, td, mask, inds_calc_by_ptr_offset, ls)
        loadexpr = Expr(:call, lv(:_vload), sptr(op), inds)
        add_memory_mask!(loadexpr, op, td, mask, ls)
        push!(loadexpr.args, falseexpr, rs) # unaligned load
        push!(q.args, Expr(:(=), mvar, loadexpr))
    elseif (u₁ > 1) & opu₁
        t = Expr(:tuple)
        sptrsym = sptr!(q, op)
        for u ∈ 1:u₁
            inds = mem_offset_u(op, td, inds_calc_by_ptr_offset, true, u-1, ls)
            loadexpr = Expr(:call, lv(:_vload), sptrsym, inds)
            domask = mask && (isvectorized(op) & ((u == u₁) | (vloopsym !== u₁loopsym)))
            add_memory_mask!(loadexpr, op, td, domask, ls)
            push!(loadexpr.args, falseexpr, rs)
            push!(t.args, loadexpr)
            # push!(q.args, Expr(:(=), mvar, loadexpr))
        end
        push!(q.args, Expr(:(=), mvar, Expr(:call, lv(:VecUnroll), t)))
    else
        inds = mem_offset_u(op, td, inds_calc_by_ptr_offset, true, 0, ls)
        loadexpr = Expr(:call, lv(:_vload), sptr(op), inds)
        add_memory_mask!(loadexpr, op, td, mask, ls)
        push!(loadexpr.args, falseexpr, rs)
        push!(q.args, Expr(:(=), mvar, loadexpr))
    end
    if isvectorized(op)
        prefetchind = prefetchisagoodidea(ls, op, td)
        iszero(prefetchind) || add_prefetches!(q, ls, op, td, prefetchind)
    elseif any(isvectorized, children(op))
        pushbroadcast!(q, mvar)
    end
    nothing
end
# function lower_load_vectorized!(
#     q::Expr, ls::LoopSet, op::Operation, td::UnrollArgs, mask::Union{Nothing,Symbol,Unsigned} = nothing
# )
#     @unpack u₁, u₁loopsym, u₂loopsym, vectorized, suffix = td
#     loopdeps = loopdependencies(op)
#     @assert isvectorized(op)
#     opu₁ = isu₁unrolled(op)
#     inds_calc_by_ptr_offset = indices_calculated_by_pointer_offsets(ls, op.ref)
#     if opu₁
#         umin = 0
#         U = u₁
#     else
#         umin = -1
#         U = 0
#     end
#     # Urange = unrolled ∈ loopdeps ? 0:U-1 : 0
#     var = variable_name(op, suffix)
#     for u ∈ umin:U-1
#         td = UnrollArgs(td, u)
#         pushvectorload!(q, op, var, td, U, vectorized, mask, opu₁, inds_calc_by_ptr_offset, reg_size(ls))
#     end
#     prefetchind = prefetchisagoodidea(ls, op, td)
#     iszero(prefetchind) || add_prefetches!(q, ls, op, td, prefetchind, umin)
#     nothing
# end
function indisvectorized(ls::LoopSet, ind::Symbol)
    for op ∈ operations(ls)
        ((op.variable === ind) && isvectorized(op)) && return true
    end
    false
end
@inline firstunroll(vu::VecUnroll) = getfield(getfield(vu,:data),1,false)
@inline firstunroll(x) = x
@inline lastunroll(vu::VecUnroll) = last(getfield(vu,:data))
@inline lastunroll(x) = x
@inline unmm(x) = x
@inline unmm(x::MM) = getfield(x, :i)
function lower_load_for_optranslation!(
    q::Expr, op::Operation, posindicator::UInt8, ls::LoopSet, td::UnrollArgs, mask::Bool, translationind::Int
)
    @unpack u₁loop, u₂loop, vloop, u₁, u₂max, suffix = td
    # @unpack u₁, u₁loopsym, u₂loopsym, vloopsym, u₂max, suffix = td
    iszero(suffix) || return
    total_unroll = u₁ + u₂max - 1
    mref = op.ref
    inds_by_ptroff = indices_calculated_by_pointer_offsets(ls, mref)
    # initial offset pointer
    # Unroll directions can be + or -
    # we want to start at minimum position.
    step₁ = gethint(step(u₁loop))
    step₂ = gethint(step(u₂loop))
    # abs of steps are equal
    equal_steps = (step₁ == step₂) ⊻ (posindicator ≠ 0x03)
    # @show step₁, step₂, posindicator, equal_steps
    # _td = UnrollArgs(u₁loop, u₂loop, vloop, total_unroll, u₂max, Core.ifelse(equal_steps, 0, u₂max - 1))
    _td = UnrollArgs(u₁loop, u₂loop, vloop, u₁, u₂max, Core.ifelse(equal_steps, 0, u₂max - 1))
    gespinds = mem_offset(op, _td, inds_by_ptroff, false, ls)
    ptr = vptr(op)
    gptr = Symbol(ptr, "##GESPED##")
    for i ∈ eachindex(gespinds.args)
        if i == translationind
            gespinds.args[i] = Expr(:call, lv(Core.ifelse(equal_steps, :firstunroll, :lastunroll)), gespinds.args[i])
        # else
        #     gespinds.args[i] = Expr(:call, lv(:unmm), gespinds.args[i])
        end
    end
    ip = GlobalRef(VectorizationBase, :increment_ptr)
    vpo = vptr_offset(gptr)
    push!(q.args, Expr(:(=), vpo, Expr(:call, ip, ptr, vptr_offset(ptr), gespinds)))
    push!(q.args, Expr(:(=), gptr, ptr))#Expr(:call, GlobalRef(VectorizationBase, :reconstruct_ptr), 
    fill!(inds_by_ptroff, true)
    @unpack ref, loopedindex = mref
    indices = copy(getindices(ref))
    # old_translation_index = indices[translationind]
    # indices[translationind] = u₁loop.itersymbol
    # @show indices, translationind, vloop
    # getindicesonly returns a view of `getindices`
    dummyref = ArrayReference(ref.array, indices, zero(getoffsets(ref)), getstrides(ref))
    # loopedindex[translationind] = true
    dummymref = ArrayReferenceMeta(dummyref, fill!(similar(loopedindex), true), gptr)
    indonly = getindicesonly(dummyref)
    for i ∈ eachindex(indonly)
        if i == translationind
            indonly[i] = u₁loop.itersymbol
        elseif !loopedindex[i]
            ind = indonly[i]
            for indop ∈ operations(ls)
                if isvectorized(indop) & (name(indop) === ind)
                    indonly[i] = vloop.itersymbol
                    break
                end
            end
        end
    end
    # @show indices
    _td = UnrollArgs(u₁loop, u₂loop, vloop, total_unroll, u₂max, -1)
    op.ref = dummymref
    # @show isu₁unrolled(op), isu₂unrolled(op)
    _lower_load!(q, ls, op, _td, mask)
    # set old values
    op.ref = mref
    # loopedindex[translationind] = false
    # indices[translationind] = old_translation_index
    shouldbroadcast = (!isvectorized(op)) && any(isvectorized, children(op))
    # now we need to assign the `Vec`s from the `VecUnroll` to the correct name.
    variable_name_u = Symbol(variable_name(op, -1), '_', total_unroll)
    variable_name_data = Symbol(variable_name_u, "##data##")
    push!(q.args, :($variable_name_data = getfield($variable_name_u, 1)))
    if shouldbroadcast
        broadcasted_data = broadcastedname(variable_name_data)
        push!(q.args, :($broadcasted_data = getfield($(broadcastedname(variable_name_u)), 1)))
    end
    gf = GlobalRef(Core,:getfield)
    for u₂ ∈ 0:u₂max-1
        variable_name_u₂ = Symbol(variable_name(op, u₂), '_', u₁)
        t = Expr(:tuple)
        if shouldbroadcast
            tb = Expr(:tuple)
        end
        for u ∈ 1:u₁
            uu = if equal_steps
                u + u₂
            else
                u - u₂ + u₂max - 1
            end
            push!(t.args, :($gf($variable_name_data, $uu)))
            if shouldbroadcast
                push!(tb.args, :($gf($broadcasted_data, $uu)))
            end
        end
        push!(q.args, Expr(:(=), variable_name_u₂, Expr(:call, lv(:VecUnroll), t)))
        if shouldbroadcast
            push!(q.args, Expr(:(=), broadcastedname(variable_name_u₂), Expr(:call, lv(:VecUnroll), tb)))
        end
    end
    nothing
end

# TODO: this code should be rewritten to be more "orthogonal", so that we're just combining separate pieces.
# Using sentinel values (eg, T = -1 for non tiling) in part to avoid recompilation.
function lower_load!(
    q::Expr, op::Operation, ls::LoopSet, td::UnrollArgs, mask::Bool
)
    @unpack u₁, u₂max, u₁loopsym, u₂loopsym, vloopsym, suffix = td
    if (suffix != -1) && ls.loadelimination
        if (u₁ > 1) & (u₂max > 1)
            istr, ispl = isoptranslation(ls, op, UnrollSymbols(u₁loopsym, u₂loopsym, vloopsym))
            if istr ≠ 0x00
                return lower_load_for_optranslation!(q, op, ispl, ls, td, mask, istr)
            end
        end
        if (suffix > 0) && (u₂loopsym !== vloopsym)
            mno, id = maxnegativeoffset(ls, op, u₂loopsym)
            if -suffix < mno < 0 # already checked that `suffix != -1` above
                varnew = variable_name(op, suffix)
                varold = variable_name(operations(ls)[id], suffix + mno)
                opold = operations(ls)[id]
                u = isu₁unrolled(op) ? u₁ : 1
                push!(q.args, Expr(:(=), Symbol(varnew, '_', u), Symbol(varold, '_', u)))
                return
            end
        end
    end
    _lower_load!(q, ls, op, td, mask)
end
function _lower_load!(
    q::Expr, ls::LoopSet, op::Operation, td::UnrollArgs, mask::Bool, inds_calc_by_ptr_offset::Vector{Bool} = indices_calculated_by_pointer_offsets(ls, op.ref)
)
  if rejectinterleave(op)
    lower_load_no_optranslation!(q, ls, op, td, mask, inds_calc_by_ptr_offset)
  else
    omop = offsetloadcollection(ls)
    @unpack opids, opidcollectionmap, batchedcollections, batchedcollectionmap = omop
    batchid, opind = batchedcollectionmap[identifier(op)]
    if opind == 1
      collectionid, copind = opidcollectionmap[identifier(op)]
      opidmap = opids[collectionid]
      idsformap = batchedcollections[batchid]
      lower_load_collection!(q, ls, opidmap, idsformap, td, mask, inds_calc_by_ptr_offset)
    end
  end
  return nothing
end
function additive_vectorized_loopinductvar_only(op::Operation)
    isvectorized(op) || return true
    isloopvalue(op) && return true
    iscompute(op) || return false
    additive_instr = (:add_fast, :(+), :vadd, :identity, :sub_fast, :(-), :vsub)
    Base.sym_in(instruction(op).instr, additive_instr) || return false
    return all(additive_vectorized_loopinductvar_only, parents(op))
end
# Checks if we cannot use `Unroll`
function rejectcurly(ls::LoopSet, op::Operation, td::UnrollArgs)
    @unpack u₁loopsym, vloopsym = td
    rejectcurly(ls, op, u₁loopsym, vloopsym)
end
function rejectcurly(ls::LoopSet, op::Operation, u₁loopsym::Symbol, vloopsym::Symbol)
    indices = getindicesonly(op)
    li = op.ref.loopedindex
    AV = AU = false
    for (n,ind) ∈ enumerate(indices)
        # @show AU, op, n, ind, vloopsym, u₁loopsym
        if li[n]
            if ind === vloopsym
                AV && return true
                AV = true
            end
            if ind === u₁loopsym
                AU && return true
                AU = true
            end
        else
            opp = findop(parents(op), ind)
            # @show opp
            if isvectorized(opp)
                AV && return true
                AV = true
            end
            if (u₁loopsym === CONSTANTZEROINDEX) ? (CONSTANTZEROINDEX ∈ loopdependencies(opp)) : (isu₁unrolled(opp))
                AU && return true
                AU = true
            end
        end
    end
    false
end
function rejectinterleave(ls::LoopSet, op::Operation, vloop::Loop, idsformap::SubArray{Tuple{Int,Int}, 1, Vector{Tuple{Int,Int}}, Tuple{UnitRange{Int}}, true})
  strd = step(vloop)
  isknown(strd) || return true
  # TODO: reject if there is a vectorized !loopedindex index
  indices = getindicesonly(op); li = op.ref.loopedindex
  for i ∈ eachindex(li)
    li[i] && continue
    ind = indices[i]
    for indop ∈ operations(ls)
      if (name(indop) === ind) && isvectorized(indop)
        additive_vectorized_loopinductvar_only(indop) || return true # so that it is `MM`
      end
    end
  end
  vloopsym = vloop.itersymbol; 
  (first(getindices(op)) === vloopsym) && (length(idsformap) ≠ first(getstrides(op)) * gethint(strd))
end
# function lower_load_collection_manual_u₁unroll!(
#     q::Expr, ls::LoopSet, opidmap::Vector{Int},
#     idsformap::SubArray{Tuple{Int,Int}, 1, Vector{Tuple{Int,Int}}, Tuple{UnitRange{Int}}, true},
#     ua::UnrollArgs, mask::Bool, inds_calc_by_ptr_offset::Vector{Bool}, op::Operation
# )
#     @unpack u₁, u₁loop, u₁loopsym, u₂loopsym, vloopsym, vloop, suffix = ua
#     _mvar = mangledvar(op)
#     op.mangledvariable = gensym!(ls,_mvar)
#     for u ∈ 0:u₁-1
#         lower_load_collection!(
#             q, ls, opidmap, idsformap, ua, mask, inds_calc_by_ptr_offset
#         )
#     end
#     op.mangledvariable = _mvar
# end
function lower_load_collection!(
    q::Expr, ls::LoopSet, opidmap::Vector{Int},
    idsformap::SubArray{Tuple{Int,Int}, 1, Vector{Tuple{Int,Int}}, Tuple{UnitRange{Int}}, true},
    ua::UnrollArgs, mask::Bool, inds_calc_by_ptr_offset::Vector{Bool}
)
    @unpack u₁, u₁loop, u₁loopsym, u₂loopsym, vloopsym, vloop, suffix = ua

    ops = operations(ls)
    nouter = length(idsformap)
    # ua = UnrollArgs(nouter, unrollsyms, u₂, 0)
    # idsformap contains (index, offset) pairs
    op = ops[opidmap[first(first(idsformap))]]
    # if isu₁unrolled(op) && u₁ > 1 && !isknown(step(u₁loop))
    #     return lower_load_collection_manual_u₁unroll!(
    #         q, ls, opidmap, idsformap, ua,
    #         mask, inds_calc_by_ptr_offset, op
    #     )
    # end
    opindices = getindices(op)
    interleave = first(opindices) === vloopsym
    # construct dummy unrolled loop
    offset_dummy_loop = Loop(first(opindices), MaybeKnown(1), MaybeKnown(1024), MaybeKnown(1), Symbol(""), Symbol(""))
    unrollcurl₂ = unrolled_curly(op, nouter, offset_dummy_loop, vloop, mask, 1) # interleave always 1 here
    inds = mem_offset_u(op, ua, inds_calc_by_ptr_offset, false, 0, ls)
    falseexpr = Expr(:call, lv(:False)); rs = staticexpr(reg_size(ls));

    opu₁, opu₂ = isunrolled_sym(op, u₁loopsym, u₂loopsym, vloopsym, ls)
    manualunrollu₁ = if opu₁ && u₁ > 1 # both unrolled
        if isknown(step(u₁loop)) && sum(Base.Fix2(===,u₁loopsym), getindicesonly(op)) == 1
            if interleave # TODO: handle this better than using `rejectinterleave`
                interleaveval = -nouter
            else
                interleaveval = 0
            end
            unrollcurl₁ = unrolled_curly(op, u₁, ua.u₁loop, vloop, mask, interleaveval)
            inds = Expr(:call, unrollcurl₁, inds)
            false
        else
            true # u₁ > 1 already checked to reach here
        end
    else
        false
    end
    uinds = Expr(:call, unrollcurl₂, inds)
    sptrsym = sptr!(q, op)
    loadexpr = Expr(:call, lv(:_vload), sptrsym, uinds)
    # not using `add_memory_mask!(storeexpr, op, ua, mask, ls)` because we checked `isconditionalmemop` earlier in `lower_load_collection!`
    u₁vectorized = u₁loopsym === vloopsym
    if (mask && isvectorized(op))
        if !(manualunrollu₁ & u₁vectorized)
            push!(loadexpr.args, MASKSYMBOL)
        end
    end
    push!(loadexpr.args, falseexpr, rs)
    collectionname = Symbol(vptr(op), "##collection##number#", opidmap[first(first(idsformap))], "#", suffix, "##size##", nouter, "##u₁##", u₁)
    gf = GlobalRef(Core,:getfield)
    if manualunrollu₁
        masklast = mask & u₁vectorized & isvectorized(op)
        extractedvs = Vector{Expr}(undef, length(idsformap))
        for i ∈ eachindex(extractedvs)
            extractedvs[i] = Expr(:tuple)
        end
        for u ∈ 0:u₁-1
            collectionname_u = Symbol(collectionname, :_, u)
            if u ≠ 0
                inds = mem_offset_u(op, ua, inds_calc_by_ptr_offset, false, u, ls)
                uinds = Expr(:call, unrollcurl₂, inds)
                loadexpr = copy(loadexpr)
                loadexpr.args[3] = Expr(:call, unrollcurl₂, inds)
                (((u+1) == u₁) & masklast) && insert!(loadexpr.args, length(loadexpr.args)-1, MASKSYMBOL) # 1 for `falseexpr` pushed at end
            end
            # unpack_collection!(q, ls, opidmap, idsformap, ua, loadexpr, collectionname, op, false)
            push!(q.args, Expr(:(=), collectionname_u, Expr(:call, gf, loadexpr, 1)))
            # getfield to extract data from `VecUnroll` object, so we have a tuple
            for (i,(opid,o)) ∈ enumerate(idsformap)
                ext = extractedvs[i]
                if (u+1) == u₁
                    _op = ops[opidmap[opid]]
                    mvar = Symbol(variable_name(_op, Core.ifelse(opu₂, suffix, -1)), '_', u₁)
                    push!(q.args, Expr(:(=), mvar, Expr(:call, lv(:VecUnroll), ext)))
                end
                push!(ext.args, Expr(:call, gf, collectionname_u, i, false))
            end
        end
    else
        push!(q.args, Expr(:(=), collectionname, Expr(:call, gf, loadexpr, 1)))
        # getfield to extract data from `VecUnroll` object, so we have a tuple
        u = Core.ifelse(opu₁, u₁, 1)
        for (i,(opid,o)) ∈ enumerate(idsformap)
            extractedv = Expr(:call, gf, collectionname, i, false)
            
            _op = ops[opidmap[opid]]
            mvar = Symbol(variable_name(_op, Core.ifelse(opu₂, suffix, -1)), '_', u)
            push!(q.args, Expr(:(=), mvar, extractedv))
        end
        # unpack_collection!(q, ls, opidmap, idsformap, ua, loadexpr, collectionname, op, true)
    end
end

