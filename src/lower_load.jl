function lower_load_scalar!(
    q::Expr, op::Operation, vectorized::Symbol, u₁loop::Symbol,
    u₂loop::Symbol, U::Int, suffix::Union{Nothing,Int}, umin::Int = 0
)
    loopdeps = loopdependencies(op)
    @assert vectorized ∉ loopdeps
    # mvar, opu₁, opu₂ = variable_name_and_unrolled(op, u₁loop, u₂loop, suffix)
    mvar = variable_name(op, suffix)
    ptr = refname(op)
    opu₁ = u₁loop ∈ loopdeps
    U = opu₁ ? U : 1
    if instruction(op).instr !== :conditionalload
        for u ∈ umin:U-1
            varname = varassignname(mvar, u, opu₁)
            td = UnrollArgs(u, u₁loop, u₂loop, suffix)
            push!(q.args, Expr(:(=), varname, Expr(:call, lv(:vload), ptr, mem_offset_u(op, td))))
        end
    else
        opu₂ = !isnothing(suffix) && u₂loop ∈ loopdeps
        condop = last(parents(op))
        condvar, condu₁ = condvarname_and_unroll(cond, u₁loop, u₂loop, suffix, opu₂)
        for u ∈ umin:U-1
            condsym = varassignname(condvar, u, condu₁)
            varname = varassignname(mvar, u, isunrolled)
            td = UnrollArgs(u, u₁loop, u₂loop, suffix)
            load = Expr(:call, lv(:vload), ptr, mem_offset_u(op, td))
            cload = Expr(:if, condsym, load, Expr(:call, :zero, Expr(:call, :eltype, ptr)))
            push!(q.args, Expr(:(=), varname, cload))
        end
    end
    nothing
end
function pushvectorload!(
    q::Expr, op::Operation, var::Symbol, td::UnrollArgs, U::Int, vectorized::Symbol, mask
)
    @unpack u, u₁loop, u₂loop, suffix = td
    ptr = refname(op)
    vecnotunrolled = vectorized !== u₁loop
    name, mo = name_memoffset(var, op, td, vecnotunrolled)
    instrcall = Expr(:call, lv(:vload), ptr, mo)

    iscondstore = instruction(op).instr === :conditionalload
    maskend = mask !== nothing && (vecnotunrolled || u == U - 1)
    if iscondstore
        condop = last(parents(op))
        opu₂ = !isnothing(suffix) && u₂loop ∈ loopdependencies(op)
        condvar, condu₁ = condvarname_and_unroll(condop, u₁loop, u₂loop, suffix, opu₂)
        condsym = varassignname(condvar, u, condu₁)
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
end
function lower_load_vectorized!(
    q::Expr, op::Operation, vectorized::Symbol, u₁loop::Symbol, u₂loop::Symbol, U::Int,
    suffix::Union{Nothing,Int}, mask::Union{Nothing,Symbol,Unsigned} = nothing, umin::Int = 0
)
    loopdeps = loopdependencies(op)
    @assert vectorized ∈ loopdeps
    if u₁loop ∈ loopdeps
        umin = umin
        U = U
    else
        umin = -1
        U = 0
    end
    # Urange = unrolled ∈ loopdeps ? 0:U-1 : 0
    var = variable_name(op, suffix)
    for u ∈ umin:U-1
        td = UnrollArgs(u, u₁loop, u₂loop, suffix)
        pushvectorload!(q, op, var, td, U, vectorized, mask)
    end
    nothing
end

# TODO: this code should be rewritten to be more "orthogonal", so that we're just combining separate pieces.
# Using sentinel values (eg, T = -1 for non tiling) in part to avoid recompilation.
function lower_load!(
    q::Expr, op::Operation, vectorized::Symbol, ls::LoopSet, u₁loop::Symbol, u₂loop::Symbol, U::Int,
    suffix::Union{Nothing,Int}, mask::Union{Nothing,Symbol,Unsigned} = nothing
)
    if !isnothing(suffix) && suffix > 0
        istr, ispl = isoptranslation(ls, op, u₁loop, u₂loop, vectorized)
        if istr && ispl
            varnew = variable_name(op, suffix)
            varold = variable_name(op, suffix - 1)
            for u ∈ 0:U-2
                push!(q.args, Expr(:(=), Symbol(varnew, u), Symbol(varold, u + 1)))
            end
            umin = U - 1
        elseif u₂loop !== vectorized
            mno, id = maxnegativeoffset(ls, op, u₂loop)
            if -suffix < mno < 0
                varnew = variable_name(op, suffix)
                varold = variable_name(operations(ls)[id], suffix + mno)
                opold = operations(ls)[id]
                if u₁loop ∈ loopdependencies(op)
                    for u ∈ 0:U-1
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
        umin = 0
    end
    if vectorized ∈ loopdependencies(op)
        lower_load_vectorized!(q, op, vectorized, u₁loop, u₂loop, U, suffix, mask, umin)
    else
        lower_load_scalar!(q, op, vectorized, u₁loop, u₂loop, U, suffix, umin)
    end
end
