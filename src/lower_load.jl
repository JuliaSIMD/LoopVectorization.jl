function lower_load_scalar!(
    q::Expr, op::Operation, ua::UnrollArgs, umin::Int, inds_calc_by_ptr_offset::Vector{Bool}
)
    loopdeps = loopdependencies(op)
    @unpack u₁, u₁loopsym, u₂loopsym, vectorized, suffix = ua
    @assert vectorized ∉ loopdeps
    # mvar, opu₁, opu₂ = variable_name_and_unrolled(op, u₁loop, u₂loop, suffix)
    mvar = variable_name(op, suffix)
    opu₁ = isu₁unrolled(op)
    ptr = vptr(op)
    U = opu₁ ? u₁ : 1
    if instruction(op).instr !== :conditionalload
        for u ∈ umin:U-1
            varname = varassignname(mvar, u, opu₁)
            td = UnrollArgs(ua, u)
            push!(q.args, Expr(:(=), varname, Expr(:call, lv(:vload), ptr, mem_offset_u(op, td, inds_calc_by_ptr_offset))))
        end
    else
        opu₂ = !isnothing(suffix) && u₂loopsym ∈ loopdeps
        condop = last(parents(op))
        condvar, condu₁ = condvarname_and_unroll(condop, u₁loopsym, u₂loopsym, suffix, opu₂)
        for u ∈ umin:U-1
            condsym = varassignname(condvar, u, condu₁)
            varname = varassignname(mvar, u, u₁loopsym ∈ loopdependencies(op))
            td = UnrollArgs(ua, u)
            load = Expr(:call, lv(:vload), ptr, mem_offset_u(op, td, inds_calc_by_ptr_offset))
            cload = Expr(:if, condsym, load, Expr(:call, :zero, Expr(:call, :eltype, ptr)))
            push!(q.args, Expr(:(=), varname, cload))
        end
    end
    nothing
end
function pushvectorload!(
    q::Expr, op::Operation, var::Symbol, td::UnrollArgs, U::Int, vectorized::Symbol, mask, u₁unrolled::Bool, inds_calc_by_ptr_offset::Vector{Bool}
)
    @unpack u₁, u₁loopsym, u₂loopsym, suffix = td
    ptr = vptr(op)
    vecnotunrolled = vectorized !== u₁loopsym
    name, mo = name_memoffset(var, op, td, u₁unrolled, inds_calc_by_ptr_offset)
    instrcall = Expr(:call, lv(:vload), ptr, mo)

    iscondstore = instruction(op).instr === :conditionalload
    maskend = mask !== nothing && (vecnotunrolled || u₁ == U - 1)
    if iscondstore
        condop = last(parents(op))
        opu₂ = !isnothing(suffix) && u₂loopsym ∈ loopdependencies(op)
        condvar, condu₁ = condvarname_and_unroll(condop, u₁loopsym, u₂loopsym, suffix, opu₂)
        condsym = varassignname(condvar, u₁, condu₁)
        if vectorized ∈ loopdependencies(condop)
            if maskend
                push!(instrcall.args, Expr(:call, :&, condsym, mask))
            else
                push!(instrcall.args, condsym)
            end
        else
            if maskend
                push!(instrcall.args, mask)
            end
            instrcall = Expr(:if, condsym, instrcall, Expr(:call, lv(:vzero), VECTORWIDTHSYMBOL, Expr(:call, :eltype, ptr)))
        end
    elseif maskend
        push!(instrcall.args, mask)
    end
    push!(q.args, Expr(:(=), name, instrcall))
    # push!(q.args, :(@show $name))
end
function prefetchisagoodidea(ls::LoopSet, op::Operation, td::UnrollArgs)
    # return false
    @unpack u₁, u₁loopsym, u₂loopsym, vectorized, u₂max, suffix = td
    length(loopdependencies(op)) ≤ 1 && return 0
    vectorized ∈ loopdependencies(op) || return 0
    u₂loopsym === Symbol("##undefined##") && return 0
    dontskip = (VectorizationBase.CACHELINE_SIZE ÷ VectorizationBase.REGISTER_SIZE) - 1
    # u₂loopsym is vectorized
    # u₁vectorized = vectorized === u₁loopsym
    u₂vectorized = vectorized === u₂loopsym
    (!isnothing(suffix) && u₂vectorized && !iszero(suffix & dontskip)) && return 0
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
    dontskip = (64 ÷ VectorizationBase.REGISTER_SIZE) - 1
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
    gespinds = mem_offset_u(op, UnrollArgs(td, 0), indices_calculated_by_pointer_offsets(ls, op.ref))
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
            if isone(u)
                inds.args[i] = Expr(:call, lv(:unwrap), VECTORWIDTHSYMBOL)
            else
                inds.args[i] = Expr(:call, lv(:valmul), VECTORWIDTHSYMBOL, u)
            end
        else
            inds.args[i] = Expr(:call, Expr(:curly, lv(:Static), u))
        end
        push!(q.args, Expr(:call, lv(:prefetch0), gptr, copy(inds)))
    end
    nothing
end
function lower_load_vectorized!(
    q::Expr, ls::LoopSet, op::Operation, td::UnrollArgs, mask::Union{Nothing,Symbol,Unsigned} = nothing, umin::Int = 0
)
    @unpack u₁, u₁loopsym, u₂loopsym, vectorized, suffix = td
    loopdeps = loopdependencies(op)
    @assert isvectorized(op)
    opu₁ = isu₁unrolled(op)
    inds_calc_by_ptr_offset = indices_calculated_by_pointer_offsets(ls, op.ref)
    if opu₁
        umin = umin
        U = u₁
    else
        umin = -1
        U = 0
    end
    # Urange = unrolled ∈ loopdeps ? 0:U-1 : 0
    var = variable_name(op, suffix)
    for u ∈ umin:U-1
        td = UnrollArgs(td, u)
        pushvectorload!(q, op, var, td, U, vectorized, mask, opu₁, inds_calc_by_ptr_offset)
    end
    prefetchind = prefetchisagoodidea(ls, op, td)
    iszero(prefetchind) || add_prefetches!(q, ls, op, td, prefetchind, umin)
    nothing
end
function indisvectorized(ls::LoopSet, ind::Symbol, vectorized::Symbol)
    for op ∈ operations(ls)
        ((op.variable === ind) && isvectorized(op)) && return true
    end
    false
end

# function lower_load_for_optranslation!(
#     q::Expr, op::Operation, ls::LoopSet, td::UnrollArgs{Int}, mask::Union{Nothing,Symbol,Unsigned}, translationind::Int
# )
#     @unpack u₁, u₁loopsym, u₂loopsym, vectorized, u₂max, suffix = td
#     iszero(suffix) || return

#     gespinds = mem_offset(op, UnrollArgs(td, 0), indices_calculated_by_pointer_offsets(ls, op.ref), false)
#     ptr = vptr(op)
#     gptr = Symbol(ptr, "##GESPED##")
#     for i ∈ eachindex(gespinds.args)
#         if i != translationind
#             gespinds.args[i] = Expr(:call, lv(:data), gespinds.args[i])
#         end
#     end    
#     push!(q.args, Expr(:(=), gptr, Expr(:call, lv(:gesp), ptr, gespinds)))

#     inds = Expr(:tuple)
#     ginds = Expr(:tuple)
#     indices = getindicesonly(op)

#     for (i,ind) ∈ enumerate(indices)
#         if i == translationind # ind cannot be the translation ind
#             push!(inds.args, Expr(:call, lv(:Zero)))
#             push!(ginds.args, Expr(:call, Expr(:curly, lv(:Static), 1)))
#         elseif (ind === vectorized) || indisvectorized(ls, ind, vectorized)
#             push!(inds.args, _MMind(Expr(:call, lv(:Zero))))
#             push!(ginds.args, Expr(:call, lv(:Zero)))
#         else
#             push!(inds.args, Expr(:call, lv(:Zero)))
#             push!(ginds.args, Expr(:call, lv(:Zero)))
#         end
#     end
#     varbase = variable_name(op, 0)
#     vloadexpr = Expr(:call, lv(:vload), gptr, inds)
#     gespexpr  = Expr(:(=), gptr, Expr(:call, lv(:gesp), gptr, ginds))
#     push!(q.args, Expr(:(=), Symbol(varbase, 0), vloadexpr))

#     for u ∈ 1:u₁-1
#         push!(q.args, gespexpr)
#         push!(q.args, Expr(:(=), Symbol(varbase, u), vloadexpr))
#     end
#     # this takes care of u₂ == 0
#     offset = u₁
#     for u₂ ∈ 1:u₂max-1
#         varold = varbase
#         varbase = variable_name(op, u₂)
#         for u ∈ 0:u₁-2
#             push!(q.args, Expr(:(=), Symbol(varbase, u), Symbol(varold, u + 1)))
#         end
#         push!(q.args, gespexpr)
#         push!(q.args, Expr(:(=), Symbol(varbase, u₁ - 1), vloadexpr))
#         offset += 1
#     end
#     nothing
# end


function lower_load_for_optranslation!(
    q::Expr, op::Operation, ls::LoopSet, td::UnrollArgs{Int}, mask::Union{Nothing,Symbol,Unsigned}, translationind::Int
)
    @unpack u₁, u₁loopsym, u₂loopsym, vectorized, u₂max, suffix = td
    iszero(suffix) || return

    gespinds = mem_offset(op, UnrollArgs(td, 0), indices_calculated_by_pointer_offsets(ls, op.ref), false)
    ptr = vptr(op)
    gptr = Symbol(ptr, "##GESPED##")
    for i ∈ eachindex(gespinds.args)
        if i != translationind
            gespinds.args[i] = Expr(:call, lv(:data), gespinds.args[i])
        end
    end    
    push!(q.args, Expr(:(=), gptr, Expr(:call, lv(:gesp), ptr, gespinds)))

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
    varbase = variable_name(op, 0)
    push!(q.args, Expr(:(=), Symbol(varbase, 0), Expr(:call, lv(:vload), gptr, copy(inds))))

    for u ∈ 1:u₁-1
        inds.args[translationind] = Expr(:call, Expr(:curly, lv(:Static), u))
        push!(q.args, Expr(:(=), Symbol(varbase, u), Expr(:call, lv(:vload), gptr, copy(inds))))
    end
    # this takes care of u₂ == 0
    offset = u₁
    for u₂ ∈ 1:u₂max-1
        varold = varbase
        varbase = variable_name(op, u₂)
        for u ∈ 0:u₁-2
            push!(q.args, Expr(:(=), Symbol(varbase, u), Symbol(varold, u + 1)))
        end
        inds.args[translationind] = Expr(:call, Expr(:curly, lv(:Static), offset))
        push!(q.args, Expr(:(=), Symbol(varbase, u₁ - 1), Expr(:call, lv(:vload), gptr, copy(inds))))
        offset += 1
    end
    nothing
end

# TODO: this code should be rewritten to be more "orthogonal", so that we're just combining separate pieces.
# Using sentinel values (eg, T = -1 for non tiling) in part to avoid recompilation.
function lower_load!(
    q::Expr, op::Operation, ls::LoopSet, td::UnrollArgs, mask::Union{Nothing,Symbol,Unsigned} = nothing
)
    @unpack u₁, u₁loopsym, u₂loopsym, vectorized, suffix = td
    if !isnothing(suffix) && ls.loadelimination[]
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
                    if isu₁unrolled(op)
                        for u ∈ 0:u₁-1
                            push!(q.args, Expr(:(=), Symbol(varnew, u), Symbol(varold, u)))
                        end
                    else
                        push!(q.args, Expr(:(=), varnew, varold))
                    end
                    return
                end
            end
        end
    end
    umin = 0
    if isvectorized(op)
        lower_load_vectorized!(q, ls, op, td, mask, umin)
    else
        lower_load_scalar!(q, op, td, umin, indices_calculated_by_pointer_offsets(ls, op.ref))
    end
end
