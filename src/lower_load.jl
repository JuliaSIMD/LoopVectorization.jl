function lower_load_scalar!(
    q::Expr, op::Operation, ua::UnrollArgs, umin::Int = 0
)
    loopdeps = loopdependencies(op)
    @unpack u₁, u₁loopsym, u₂loopsym, vectorized, suffix = ua
    @assert vectorized ∉ loopdeps
    # mvar, opu₁, opu₂ = variable_name_and_unrolled(op, u₁loop, u₂loop, suffix)
    mvar = variable_name(op, suffix)
    opu₁ = u₁loopsym ∈ loopdeps
    unrolled = (opu₁ || u₂loopsym ∈ loopdeps)
    ptr = unrolled ? offset_refname(op) : refname(op)
    U = opu₁ ? u₁ : 1
    if instruction(op).instr !== :conditionalload
        for u ∈ umin:U-1
            varname = varassignname(mvar, u, opu₁)
            td = UnrollArgs(ua, u)
            push!(q.args, Expr(:(=), varname, Expr(:call, lv(:vload), ptr, mem_offset_u(op, td, unrolled))))
        end
    else
        opu₂ = !isnothing(suffix) && u₂loopsym ∈ loopdeps
        condop = last(parents(op))
        condvar, condu₁ = condvarname_and_unroll(condop, u₁loopsym, u₂loopsym, suffix, opu₂)
        for u ∈ umin:U-1
            condsym = varassignname(condvar, u, condu₁)
            varname = varassignname(mvar, u, u₁loopsym ∈ loopdependencies(op))
            td = UnrollArgs(ua, u)
            load = Expr(:call, lv(:vload), ptr, mem_offset_u(op, td, unrolled))
            cload = Expr(:if, condsym, load, Expr(:call, :zero, Expr(:call, :eltype, ptr)))
            push!(q.args, Expr(:(=), varname, cload))
        end
    end
    nothing
end
function pushvectorload!(
    q::Expr, op::Operation, var::Symbol, td::UnrollArgs, U::Int, vectorized::Symbol, mask, u₁unrolled::Bool, unrolled::Bool
)
    @unpack u₁, u₁loopsym, u₂loopsym, suffix = td
    ptr = unrolled ? offset_refname(op) : refname(op)
    vecnotunrolled = vectorized !== u₁loopsym
    name, mo = name_memoffset(var, op, td, u₁unrolled, unrolled)
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
    @unpack u₁, u₁loopsym, u₂loopsym, vectorized, suffix = td
    vectorized ∈ loopdependencies(op) || return 0
    u₂loopsym === Symbol("##undefined##") && return 0
    dontskip = (64 ÷ VectorizationBase.REGISTER_SIZE) - 1
    (!isnothing(suffix) && (vectorized === u₂loopsym) && !iszero(suffix & dontskip)) && return 0
    innermostloopsym = last(ls.loop_order.bestorder)
    loopedindex = op.ref.loopedindex
    if length(loopedindex) > 1 && first(loopedindex)
        indices = getindices(op)
        if first(indices) === vectorized && last(indices) === innermostloopsym
            innermostloopindv = findall(map(isequal(innermostloopsym), getindices(op)))
            isone(length(innermostloopindv)) || return 0
            innermostloopind = first(innermostloopindv)
            if prod(s -> length(getloop(ls, s)), @view(indices[1:innermostloopind-1])) ≥ 120 && length(getloop(ls, innermostloopind)) ≥ 120
                if op.ref.ref.offsets[innermostloopind] < 120
                    for opp ∈ operations(ls)
                        iscompute(opp) && load_constrained(opp, u₁loopsym, u₂loopsym) && return 0
                    end
                    return innermostloopind
                end
            end
        end
    end
    0
end
function lower_load_vectorized!(
    q::Expr, ls::LoopSet, op::Operation, td::UnrollArgs, mask::Union{Nothing,Symbol,Unsigned} = nothing, umin::Int = 0
)
    @unpack u₁, u₁loopsym, u₂loopsym, vectorized, suffix = td
    loopdeps = loopdependencies(op)
    @assert vectorized ∈ loopdeps
    opu₁ = u₁loopsym ∈ loopdeps
    unrolled = opu₁ || u₂loopsym ∈ loopdeps
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
        pushvectorload!(q, op, var, td, U, vectorized, mask, opu₁, unrolled)
    end
    prefetchind = prefetchisagoodidea(ls, op, td)
    if !iszero(prefetchind)
        dontskip = (64 ÷ VectorizationBase.REGISTER_SIZE) - 1
        ptr = offset_refname(op)
        innermostloopsym = last(ls.loop_order.bestorder)
        us = ls.unrollspecification[]
        prefetch_multiplier = 4
        prefetch_distance = u₁loopsym === innermostloopsym ? us.u₁ : ( u₂loopsym === innermostloopsym ? us.u₂ : 1 )
        prefetch_distance *= prefetch_multiplier
        offsets = op.ref.ref.offsets
        inner_offset = offsets[prefetchind]
        for u ∈ umin:U-1
        # for u ∈ umin:min(umin,U-1)
            (u₁loopsym === vectorized && !iszero(u & dontskip)) && continue
            offsets[prefetchind] = inner_offset + prefetch_distance
            mo = mem_offset_u(op, UnrollArgs(td, u), true)
            instrcall = Expr(:call, lv(:prefetch0), ptr, mo)
            push!(q.args, instrcall)
        end
        offsets[prefetchind] = inner_offset
    end
    nothing
end

# TODO: this code should be rewritten to be more "orthogonal", so that we're just combining separate pieces.
# Using sentinel values (eg, T = -1 for non tiling) in part to avoid recompilation.
function lower_load!(
    q::Expr, op::Operation, ls::LoopSet, td::UnrollArgs, mask::Union{Nothing,Symbol,Unsigned} = nothing
)
    @unpack u₁, u₁loopsym, u₂loopsym, vectorized, suffix = td
    if !isnothing(suffix) && suffix > 0
        istr, ispl = isoptranslation(ls, op, UnrollSymbols(u₁loopsym, u₂loopsym, vectorized))
        if istr && ispl
            varnew = variable_name(op, suffix)
            varold = variable_name(op, suffix - 1)
            for u ∈ 0:u₁-2
                push!(q.args, Expr(:(=), Symbol(varnew, u), Symbol(varold, u + 1)))
            end
            umin = u₁ - 1
        elseif u₂loopsym !== vectorized
            mno, id = maxnegativeoffset(ls, op, u₂loopsym)
            if -suffix < mno < 0
                varnew = variable_name(op, suffix)
                varold = variable_name(operations(ls)[id], suffix + mno)
                opold = operations(ls)[id]
                if u₁loopsym ∈ loopdependencies(op)
                    for u ∈ 0:u₁-1
                        push!(q.args, Expr(:(=), Symbol(varnew, u), Symbol(varold, u)))
                    end
                else
                    push!(q.args, Expr(:(=), varnew, varold))
                end
                return
            else
                umin = 0
            end
        else
            umin = 0
        end
    else
        maybegesp_call!(q, op, td)
        umin = 0
    end
    if vectorized ∈ loopdependencies(op)
        lower_load_vectorized!(q, ls, op, td, mask, umin)
    else
        lower_load_scalar!(q, op, td, umin)
    end
end
