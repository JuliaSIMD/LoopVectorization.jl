function prefetchisagoodidea(ls::LoopSet, op::Operation, td::UnrollArgs)
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
            innermostloopindv = findall(map(isequal(innermostloopsym), getindices(op)))
            isone(length(innermostloopindv)) || return 0
            innermostloopind = first(innermostloopindv)
            if prod(s -> Float64(looplength(ls, s)), @view(indices[1:innermostloopind-1])) ≥ 120.0 && length(getloop(ls, innermostloopsym)) ≥ 120
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
    @unpack u₁, u₁loopsym, u₂loopsym, vloopsym, u₂max = td
    # we should only be here if `unitsride(vloop)`
    dontskip = (cache_lnsze(ls) ÷ reg_size(ls)) - 1
    ptr = vptr(op)
    innermostloopsym = first(names(ls))
    us = ls.unrollspecification[]
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
    gespinds = mem_offset_u(op, td, indices_calculated_by_pointer_offsets(ls, op.ref), false, 0)
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
    push!(q.args, Expr(:(=), gptr, Expr(:call, lv(:gesp), ptr, gespinds)))

    inds = Expr(:tuple)
    indices = getindicesonly(op)

    i = 0
    for (j,ind) ∈ enumerate(indices)
        push!(inds.args, Expr(:call, lv(:Zero)))
        (ind == u₁loopsym) && (i = j)
    end
    push!(q.args, Expr(:call, lv(:prefetch0), gptr, copy(inds)))
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
        push!(q.args, Expr(:call, lv(:prefetch0), gptr, copy(inds)))
    end
    nothing
end
function pushbroadcast!(q::Expr, mvar::Symbol)
    push!(q.args, Expr(:(=), Symbol(mvar, "##broadcasted##"), Expr(:call, lv(:vbroadcast), VECTORWIDTHSYMBOL, mvar)))
end
function lower_load_no_optranslation!(
    q::Expr, ls::LoopSet, op::Operation, td::UnrollArgs, mask::Bool, inds_calc_by_ptr_offset::Vector{Bool}
)
    @unpack u₁, u₁loopsym, u₂loopsym, vloopsym, suffix = td
    loopdeps = loopdependencies(op)
    # @assert isvectorized(op)
    opu₁ = isu₁unrolled(op)

    u = ifelse(opu₁, u₁, 1)
    mvar = Symbol(variable_name(op, Core.ifelse(isu₂unrolled(op), suffix,-1)), '_', u)
    falseexpr = Expr(:call, lv(:False)); rs = staticexpr(reg_size(ls))

    if all(op.ref.loopedindex)
        inds = unrolledindex(op, td, mask, inds_calc_by_ptr_offset)
        loadexpr = Expr(:call, lv(:vload), vptr(op), inds)
        add_memory_mask!(loadexpr, op, td, mask)
        push!(loadexpr.args, falseexpr, rs) # unaligned load
        push!(q.args, Expr(:(=), mvar, loadexpr))
    elseif u₁ > 1
        # t = Expr(:tuple)
        # for u ∈ 1:u₁
        let t = u₁, t = q
            inds = mem_offset_u(op, td, inds_calc_by_ptr_offset, true, u-1)
            loadexpr = Expr(:call, lv(:vload), vptr(op), inds)
            add_memory_mask!(loadexpr, op, td, mask & ((u == u₁) | isvectorized(op)))
            push!(loadexpr.args, falseexpr, rs)
            # push!(t.args, loadexpr)
            push!(t.args, Expr(:(=), mvar, loadexpr))
        end
        # push!(q.args, Expr(:(=), mvar, Expr(:call, lv(:VecUnroll), t)))
    else
        inds = mem_offset_u(op, td, inds_calc_by_ptr_offset, true, 0)
        loadexpr = Expr(:call, lv(:vload), vptr(op), inds)
        add_memory_mask!(loadexpr, op, td, mask)
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
function lower_load_for_optranslation!(
    q::Expr, op::Operation, ls::LoopSet, td::UnrollArgs, mask::Bool, translationind::Int
)
    @unpack u₁, u₁loopsym, u₂loopsym, vloopsym, u₂max, suffix = td
    iszero(suffix) || return

    # initial offset pointer
    gespinds = mem_offset(op, UnrollArgs(td, u₁), indices_calculated_by_pointer_offsets(ls, op.ref), false)
    ptr = vptr(op)
    gptr = Symbol(ptr, "##GESPED##")
    for i ∈ eachindex(gespinds.args)
        if i == translationind
            gespinds.args[i] = Expr(:call, lv(:firstunroll), gespinds.args[i])
        else
            gespinds.args[i] = Expr(:call, lv(:data), gespinds.args[i])
        end
    end
    push!(q.args, Expr(:(=), gptr, Expr(:call, lv(:gesp), ptr, gespinds)))

    shouldbroadcast = (!isvectorized(op)) && any(isvectorized, children(op))

    inds = Expr(:tuple)
    indices = getindicesonly(op)
    for (i,ind) ∈ enumerate(indices)
        if i == translationind # vectorized ind cannot be the translation ind
            push!(inds.args, Expr(:call, Expr(:curly, lv(:Static), 0)))
        elseif (ind === vloopsym) || indisvectorized(ls, ind)
            push!(inds.args, _MMind(Expr(:call, lv(:Zero))))
        else
            push!(inds.args, Expr(:call, lv(:Zero)))
        end
    end
    variable_name0 = variable_name(op, 0)
    varbase = Symbol("##var##", variable_name0)
    loadcall = Expr(:call, lv(:vload), gptr, copy(inds))
    falseexpr = Expr(:call, lv(:False)); rs = staticexpr(reg_size(ls));
    mask && push!(loadcall.args, MASKSYMBOL)
    push!(loadcall.args, falseexpr, rs)

    varbase0 = Symbol(varbase, 0)
    t = Expr(:tuple, varbase0)
    push!(q.args, Expr(:(=), varbase0, loadcall))
    for u ∈ 1:u₁-1
        inds.args[translationind] = Expr(:call, Expr(:curly, lv(:Static), u))
        loadcall = Expr(:call, lv(:vload), gptr, copy(inds))
        mask && push!(loadcall.args, MASKSYMBOL)
        push!(loadcall.args, falseexpr, rs)
        varbaseu = Symbol(varbase, u)
        push!(q.args, Expr(:(=), varbaseu, loadcall))
        push!(t.args, varbaseu)
    end
    vecunroll_name = Symbol(variable_name0, '_', u₁)
    push!(q.args, Expr(:(=), vecunroll_name, Expr(:call, lv(:VecUnroll), t)))
    shouldbroadcast && pushbroadcast!(q, vecunroll_name)
    # this takes care of u₂ == 0
    offset = u₁
    for u₂ ∈ 1:u₂max-1
        t = Expr(:tuple)
        varold = varbase
        varbase = variable_name(op, u₂)
        for u ∈ 0:u₁-2
            varbaseu = Symbol(varbase, u)
            push!(q.args, Expr(:(=), varbaseu, Symbol(varold, u + 1)))
            push!(t.args, varbaseu)
        end
        inds.args[translationind] = Expr(:call, Expr(:curly, lv(:Static), offset))
        loadcall = Expr(:call, lv(:vload), gptr, copy(inds))
        mask && push!(loadcall.args, MASKSYMBOL)
        push!(loadcall.args, falseexpr, rs)
        varload = Symbol(varbase, u₁ - 1)
        push!(q.args, Expr(:(=), varload, loadcall))
        push!(t.args, varload)
        offset += 1
        vecunroll_name = Symbol(variable_name(op, u₂), '_', u₁)
        push!(q.args, Expr(:(=), vecunroll_name, Expr(:call, lv(:VecUnroll), t)))
        shouldbroadcast && pushbroadcast!(q, vecunroll_name)
    end
    nothing
end

# TODO: this code should be rewritten to be more "orthogonal", so that we're just combining separate pieces.
# Using sentinel values (eg, T = -1 for non tiling) in part to avoid recompilation.
function lower_load!(
    q::Expr, op::Operation, ls::LoopSet, td::UnrollArgs, mask::Bool
)
    @unpack u₁, u₁loopsym, u₂loopsym, vloopsym, suffix = td
    if (suffix != -1) && ls.loadelimination[]
        istr, ispl = isoptranslation(ls, op, UnrollSymbols(u₁loopsym, u₂loopsym, vloopsym))
        if !iszero(istr) & ispl
            return lower_load_for_optranslation!(q, op, ls, td, mask, istr)
        elseif suffix > 0
            if u₂loopsym !== vloopsym
                mno, id = maxnegativeoffset(ls, op, u₂loopsym)
                if -suffix < mno < 0 # already checked that `suffix != -1` above
                    varnew = variable_name(op, suffix)
                    varold = variable_name(operations(ls)[id], suffix + mno)
                    opold = operations(ls)[id]
                    u = isu₁unrolled(op) ? u₁ : 1
                    push!(q.args, Expr(:(=), Symbol(varnew, '_', u), Symbol(varold, '_', u)))
                    # if isu₁unrolled(op)
                    #     for u ∈ 0:u₁-1
                    #         push!(q.args, Expr(:(=), Symbol(varnew, u), Symbol(varold, u)))
                    #     end
                    # else

                    # end
                    return
                end
            end
        end
    end
    _lower_load!(q, ls, op, td, mask)
end
function _lower_load!(
    q::Expr, ls::LoopSet, op::Operation, td::UnrollArgs, mask::Bool
)
    omop = offsetloadcollection(ls)
    batchid, opind = omop.batchedcollectionmap[identifier(op)]
    inds_calc_by_ptr_offset = indices_calculated_by_pointer_offsets(ls, op.ref)
    # @show batchid == 0 (!isvectorized(op)) rejectinterleave(op, td.vloop, idsformap)
    if batchid == 0 || (!isvectorized(op)) || (rejectinterleave(op, td.vloop, omop.batchedcollections[batchid]))
        lower_load_no_optranslation!(q, ls, op, td, mask, inds_calc_by_ptr_offset)
    elseif opind == 1# only lower loads once
        # I do not believe it is possible for `opind == 1` to be lowered after an  operation depending on a different opind.
        # lower_load_collection!(q, ls, op, td, mask, collectionid)
        omop = offsetloadcollection(ls)
        collectionid, copind = omop.opidcollectionmap[identifier(op)]
        opidmap = offsetloadcollection(ls).opids[collectionid]
        idsformap = omop.batchedcollections[batchid]
        lower_load_collection!(q, ls, opidmap, idsformap, td, mask, inds_calc_by_ptr_offset)
    end
end
function rejectinterleave(op::Operation, vloop::Loop, idsformap::SubArray{Tuple{Int,Int}, 1, Vector{Tuple{Int,Int}}, Tuple{UnitRange{Int}}, true})
    vloopsym = vloop.itersymbol; strd = step(vloop)
    isknown(strd) || return true
    (first(getindices(op)) === vloopsym) && (length(idsformap) ≠ first(getstrides(op)) * gethint(strd))
end
function lower_load_collection!(
    q::Expr, ls::LoopSet, opidmap::Vector{Int},
    idsformap::SubArray{Tuple{Int,Int}, 1, Vector{Tuple{Int,Int}}, Tuple{UnitRange{Int}}, true},
    ua::UnrollArgs, mask::Bool, inds_calc_by_ptr_offset::Vector{Bool}
)
    @unpack u₁, u₁loopsym, u₂loopsym, vloopsym, vloop, suffix = ua
    ops = operations(ls)
    nouter = length(idsformap)
    # ua = UnrollArgs(nouter, unrollsyms, u₂, 0)
    # idsformap contains (index, offset) pairs
    op = ops[opidmap[first(first(idsformap))]]
    opindices = getindices(op)
    interleave = first(opindices) === vloopsym
    # construct dummy unrolled loop 
    offset_dummy_loop = Loop(first(opindices), MaybeKnown(1), MaybeKnown(1024), MaybeKnown(1), Symbol(""), Symbol(""))
    unrollcurl₂ = unrolled_curly(op, nouter, offset_dummy_loop, vloop, mask, 1) # interleave always 1 here
    inds = mem_offset_u(op, ua, inds_calc_by_ptr_offset, false)
    falseexpr = Expr(:call, lv(:False)); rs = staticexpr(reg_size(ls));
    if isu₁unrolled(op) && u₁ > 1 # both unrolled
        if interleave # TODO: handle this better than using `rejectinterleave`
            interleaveval = -nouter
        else
            interleaveval = 0
        end
        unrollcurl₁ = unrolled_curly(op, u₁, ua.u₁loop, vloop, mask, interleaveval)
        inds = Expr(:call, unrollcurl₁, inds)
    end
    uinds = Expr(:call, unrollcurl₂, inds)
    vp = vptr(op)
    loadexpr = Expr(:call, lv(:vload), vp, uinds)
    # not using `add_memory_mask!(storeexpr, op, ua, mask)` because we checked `isconditionalmemop` earlier in `lower_load_collection!`
    (mask && isvectorized(op)) && push!(loadexpr.args, MASKSYMBOL)
    push!(loadexpr.args, falseexpr, rs)
    collectionname = Symbol(vp, "##collection##number", first(first(idsformap)), "##size##", nouter, "##u₁##", u₁)
    # getfield to extract data from `VecUnroll` object, so we have a tuple
    push!(q.args, Expr(:(=), collectionname, Expr(:call, :getfield, loadexpr, 1)))
    u = Core.ifelse(isu₁unrolled(op), u₁, 1)
    for (i,(opid,o)) ∈ enumerate(idsformap)
        _op = ops[opidmap[opid]]
        mvar = Symbol(variable_name(_op, Core.ifelse(isu₂unrolled(_op), suffix, -1)), '_', u)
        push!(q.args, Expr(:(=), mvar, Expr(:call, :getfield, collectionname, i, false)))
    end
end
