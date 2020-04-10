function lower_load_scalar!(
    q::Expr, op::Operation, vectorized::Symbol, W::Symbol, unrolled::Symbol,
    tiled::Symbol, U::Int, suffix::Union{Nothing,Int}, umin::Int = 0
)
    loopdeps = loopdependencies(op)
    @assert vectorized ∉ loopdeps
    var = variable_name(op, suffix)
    ptr = refname(op)
    isunrolled = unrolled ∈ loopdeps
    U = isunrolled ? U : 1
    if instruction(op).instr !== :conditionalload
        for u ∈ umin:U-1
            varname = varassignname(var, u, isunrolled)
            td = UnrollArgs(u, unrolled, tiled, suffix)
            push!(q.args, Expr(:(=), varname, Expr(:call, lv(:vload), ptr, mem_offset_u(op, td))))
        end
    else
        condop = last(parents(op))
        condvar = tiled ∈ loopdependencies(condop) ? variable_name(condop, suffix) : variable_name(condop, nothing)
        condunrolled = unrolled ∈ loopdependencies(condop)
        for u ∈ umin:U-1
            condsym = condunrolled ? Symbol(condvar, u) : condvar
            varname = varassignname(var, u, isunrolled)
            td = UnrollArgs(u, unrolled, tiled, suffix)
            load = Expr(:call, lv(:vload), ptr, mem_offset_u(op, td))
            cload = Expr(:if, condsym, load, Expr(:call, :zero, Expr(:call, :eltype, ptr)))
            push!(q.args, Expr(:(=), varname, cload))
        end
    end
    nothing
end
function pushvectorload!(
    q::Expr, op::Operation, var::Symbol, td::UnrollArgs, U::Int, W::Symbol, vectorized::Symbol, mask
)
    @unpack u, unrolled, tiled, suffix = td
    ptr = refname(op)
    vecnotunrolled = vectorized !== unrolled
    name, mo = name_memoffset(var, op, td, W, vecnotunrolled)
    instrcall = Expr(:call, lv(:vload), ptr, mo)

    iscondstore = instruction(op).instr === :conditionalload
    maskend = mask !== nothing && (vecnotunrolled || u == U - 1)
    if iscondstore
        condop = last(parents(op))
        # @show condop
        condsym = tiled ∈ loopdependencies(condop) ? variable_name(condop, suffix) : variable_name(condop, nothing)
        condsym = unrolled ∈ loopdependencies(condop) ? Symbol(condsym, u) : condsym        
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
            instrcall = Expr(:if, condsym, instrcall, Expr(:call, lv(:vzero), W, Expr(:call, :eltype, ptr)))
        end
    elseif maskend
        push!(instrcall.args, mask)
    end
    push!(q.args, Expr(:(=), name, instrcall))
end
function lower_load_vectorized!(
    q::Expr, op::Operation, vectorized::Symbol, W::Symbol, unrolled::Symbol, tiled::Symbol, U::Int,
    suffix::Union{Nothing,Int}, mask::Union{Nothing,Symbol,Unsigned} = nothing, umin::Int = 0
)
    loopdeps = loopdependencies(op)
    @assert vectorized ∈ loopdeps
    if unrolled ∈ loopdeps
        umin = umin
        U = U
    else
        umin = -1
        U = 0
    end
    # Urange = unrolled ∈ loopdeps ? 0:U-1 : 0
    var = variable_name(op, suffix)
    for u ∈ umin:U-1
        td = UnrollArgs(u, unrolled, tiled, suffix)
        pushvectorload!(q, op, var, td, U, W, vectorized, mask)
    end
    nothing
end

# TODO: this code should be rewritten to be more "orthogonal", so that we're just combining separate pieces.
# Using sentinel values (eg, T = -1 for non tiling) in part to avoid recompilation.
function lower_load!(
    q::Expr, op::Operation, vectorized::Symbol, ls::LoopSet, unrolled::Symbol, tiled::Symbol, U::Int,
    suffix::Union{Nothing,Int}, mask::Union{Nothing,Symbol,Unsigned} = nothing
)
    if !isnothing(suffix) && suffix > 0
        istr, ispl = isoptranslation(ls, op, unrolled, tiled, vectorized)
        if istr && ispl
            varnew = variable_name(op, suffix)
            varold = variable_name(op, suffix - 1)
            for u ∈ 0:U-2
                push!(q.args, Expr(:(=), Symbol(varnew, u), Symbol(varold, u + 1)))
            end
            umin = U - 1
        elseif tiled !== vectorized
            mno, id = maxnegativeoffset(ls, op, tiled)
            if -suffix < mno < 0
                varnew = variable_name(op, suffix)
                varold = variable_name(operations(ls)[id], suffix + mno)
                opold = operations(ls)[id]
                if unrolled ∈ loopdependencies(op)
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
    W = ls.W
    if vectorized ∈ loopdependencies(op)
        lower_load_vectorized!(q, op, vectorized, W, unrolled, tiled, U, suffix, mask, umin)
    else
        lower_load_scalar!(q, op, vectorized, W, unrolled, tiled, U, suffix, umin)
    end
end
