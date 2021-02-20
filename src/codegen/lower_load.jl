function prefetchisagoodidea(ls::LoopSet, op::Operation, td::UnrollArgs)
    # return false
    @unpack u₁, u₁loopsym, u₂loopsym, vectorized, u₂max, suffix = td
    length(loopdependencies(op)) ≤ 1 && return 0
    vectorized ∈ loopdependencies(op) || return 0
    u₂loopsym === Symbol("##undefined##") && return 0
    # @show cache_lnsze(ls) reg_size(ls) pointer_from_objref(ls.register_size)
    dontskip = (cache_lnsze(ls) ÷ reg_size(ls)) - 1
    # u₂loopsym is vectorized
    # u₁vectorized = vectorized === u₁loopsym
    u₂vectorized = vectorized === u₂loopsym
    ((suffix != -1) && u₂vectorized && !iszero(suffix & dontskip)) && return 0
    innermostloopsym = first(names(ls))
    loopedindex = op.ref.loopedindex
    if length(loopedindex) > 1 && first(loopedindex)
        indices = getindices(op)
        if first(indices) === vectorized && first(indices) !== innermostloopsym
            # We want at least 4 reuses per load
            uses = ifelse(isu₁unrolled(op), 1, u₁)
            uses = ifelse(isu₂unrolled(op), uses, uses * u₂max)
            uses < 4 && return 0
            innermostloopindv = findall(map(isequal(innermostloopsym), getindices(op)))
            isone(length(innermostloopindv)) || return 0
            innermostloopind = first(innermostloopindv)
            if prod(s -> Float64(looplength(ls, s)), @view(indices[1:innermostloopind-1])) ≥ 120.0 && length(getloop(ls, innermostloopsym)) ≥ 120
                if op.ref.ref.offsets[innermostloopind] < 120
                    for opp ∈ operations(ls)
                        iscompute(opp) && (innermostloopsym ∈ loopdependencies(opp)) && load_constrained(opp, u₁loopsym, u₂loopsym, innermostloopsym, true) && return 0
                    end
                    return innermostloopind
                end
            end
        end
    end
    0
end
function add_prefetches!(q::Expr, ls::LoopSet, op::Operation, td::UnrollArgs, prefetchind::Int, umin::Int)
    @unpack u₁, u₁loopsym, u₂loopsym, vectorized, u₂max = td
    dontskip = (64 ÷ reg_size(ls)) - 1
    ptr = vptr(op)
    innermostloopsym = first(names(ls))
    us = ls.unrollspecification[]
    prefetch_distance = u₁loopsym === innermostloopsym ? us.u₁ : ( u₂loopsym === innermostloopsym ? us.u₂ : 1 )
    # prefetch_distance = u₁loopsym === innermostloopsym ? u₁ : ( u₂loopsym === innermostloopsym ? u₂max : 1 )
    prefetch_multiplier = 5
    prefetch_distance *= prefetch_multiplier
    offsets = op.ref.ref.offsets
    inner_offset = offsets[prefetchind]
    offsets[prefetchind] = inner_offset + prefetch_distance
    gespinds = mem_offset_u(op, td, indices_calculated_by_pointer_offsets(ls, op.ref), false, 0)
    offsets[prefetchind] = inner_offset
    ptr = vptr(op)
    gptr = Symbol(ptr, "##GESPEDPREFETCH##")
    for i ∈ eachindex(gespinds.args) 
       gespinds.args[i] = Expr(:call, lv(:data), gespinds.args[i])
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

    for u ∈ 1+umin:u₁
        # for u ∈ umin:min(umin,U-1)
        (u₁loopsym === vectorized && !iszero(u & dontskip)) && continue
        if u₁loopsym === vectorized
            # W = ls.vector_width[]
            # if W != 0
            #     inds.args[i] = staticexpr(W*u)
            # else
            if isone(u)
                inds.args[i] = VECTORWIDTHSYMBOL
            else
                inds.args[i] = Expr(:call, lv(:vmul_fast), VECTORWIDTHSYMBOL, staticexpr(u))
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
    q::Expr, ls::LoopSet, op::Operation, td::UnrollArgs, mask::Bool
)
    @unpack u₁, u₁loopsym, u₂loopsym, vectorized, suffix = td
    loopdeps = loopdependencies(op)
    # @assert isvectorized(op)
    opu₁ = isu₁unrolled(op)
    inds_calc_by_ptr_offset = indices_calculated_by_pointer_offsets(ls, op.ref)
    u = ifelse(opu₁, u₁, 1)
    mvar = Symbol(variable_name(op, suffix), '_', u)
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
        umin = isu₁unrolled(op) - 1
        iszero(prefetchind) || add_prefetches!(q, ls, op, td, prefetchind, umin)
    elseif any(isvectorized, children(op))
        pushbroadcast!(q, mvar)
    end
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
function indisvectorized(ls::LoopSet, ind::Symbol, vectorized::Symbol)
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
    @unpack u₁, u₁loopsym, u₂loopsym, vectorized, u₂max, suffix = td
    iszero(suffix) || return

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
        if i == translationind # ind cannot be the translation ind
            push!(inds.args, Expr(:call, Expr(:curly, lv(:Static), 0)))
        elseif (ind === vectorized) || indisvectorized(ls, ind, vectorized)
            push!(inds.args, _MMind(Expr(:call, lv(:Zero))))
        else
            push!(inds.args, Expr(:call, lv(:Zero)))
        end
    end
    variable_name0 = variable_name(op, 0)
    varbase = Symbol("##var##", variable_name0)
    loadcall = Expr(:call, lv(:vload), gptr, copy(inds))
    falseexpr = Expr(:call, lv(:False)); rs = staticexpr(reg_size(ls));
    mask ? push!(loadcall.args, MASKSYMBOL, falseexpr, rs) : push!(loadcall.args, falseexpr, rs)

    varbase0 = Symbol(varbase, 0)
    t = Expr(:tuple, varbase0)    
    push!(q.args, Expr(:(=), varbase0, loadcall))
    for u ∈ 1:u₁-1
        inds.args[translationind] = Expr(:call, Expr(:curly, lv(:Static), u))
        loadcall = Expr(:call, lv(:vload), gptr, copy(inds))
        mask ? push!(loadcall.args, MASKSYMBOL, falseexpr, rs) : push!(loadcall.args, falseexpr, rs)
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
        mask ? push!(loadcall.args, MASKSYMBOL, falseexpr, rs) : push!(loadcall.args, falseexpr, rs)
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
    @unpack u₁, u₁loopsym, u₂loopsym, vectorized, suffix = td
    if (suffix != -1) && ls.loadelimination[]
        istr, ispl = isoptranslation(ls, op, UnrollSymbols(u₁loopsym, u₂loopsym, vectorized))
        if !iszero(istr) & ispl
            return lower_load_for_optranslation!(q, op, ls, td, mask, istr)
        elseif suffix > 0
            if u₂loopsym !== vectorized
                mno, id = maxnegativeoffset(ls, op, u₂loopsym)
                if -suffix < mno < 0
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
    lower_load_no_optranslation!(q, ls, op, td, mask)
end
