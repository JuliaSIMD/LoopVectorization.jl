function pushvectorload!(q::Expr, op::Operation, var::Symbol, td::UnrollArgs, U::Int, W::Symbol, mask, vecnotunrolled::Bool)
    @unpack u, unrolled = td
    ptr = refname(op)
    name, mo = name_memoffset(var, op, td, W, vecnotunrolled)
    instrcall = Expr(:call, lv(:vload), ptr, mo)
    if mask !== nothing && (vecnotunrolled || u == U - 1)
        push!(instrcall.args, mask)
    end
    push!(q.args, Expr(:(=), name, instrcall))
end
function lower_load_scalar!( 
    q::Expr, op::Operation, vectorized::Symbol, W::Symbol, unrolled::Symbol, tiled::Symbol, U::Int,
    suffix::Union{Nothing,Int}, mask::Union{Nothing,Symbol,Unsigned} = nothing
)
    loopdeps = loopdependencies(op)
    @assert vectorized ∉ loopdeps
    var = variable_name(op, suffix)
    ptr = refname(op)
    isunrolled = unrolled ∈ loopdeps
    U = isunrolled ? U : 1
    for u ∈ 0:U-1
        varname = varassignname(var, u, isunrolled)
        td = UnrollArgs(u, unrolled, tiled, suffix)
        push!(q.args, Expr(:(=), varname, Expr(:call, lv(:vload), ptr, mem_offset_u(op, td))))
    end
    nothing
end
function lower_load_vectorized!(
    q::Expr, op::Operation, vectorized::Symbol, W::Symbol, unrolled::Symbol, tiled::Symbol, U::Int,
    suffix::Union{Nothing,Int}, mask::Union{Nothing,Symbol,Unsigned} = nothing
)
    loopdeps = loopdependencies(op)
    @assert vectorized ∈ loopdeps
    if unrolled ∈ loopdeps
        umin = 0
        U = U
    else
        umin = -1
        U = 0
    end
    # Urange = unrolled ∈ loopdeps ? 0:U-1 : 0
    var = variable_name(op, suffix)
    vecnotunrolled = vectorized !== unrolled
    for u ∈ umin:U-1
        td = UnrollArgs(u, unrolled, tiled, suffix)
        pushvectorload!(q, op, var, td, U, W, mask, vecnotunrolled)
    end
    nothing
end

# TODO: this code should be rewritten to be more "orthogonal", so that we're just combining separate pieces.
# Using sentinel values (eg, T = -1 for non tiling) in part to avoid recompilation.
function lower_load!(
    q::Expr, op::Operation, vectorized::Symbol, W::Symbol, unrolled::Symbol, tiled::Symbol, U::Int,
    suffix::Union{Nothing,Int}, mask::Union{Nothing,Symbol,Unsigned} = nothing
)
    if vectorized ∈ loopdependencies(op)
        lower_load_vectorized!(q, op, vectorized, W, unrolled, tiled, U, suffix, mask)
    else
        lower_load_scalar!(q, op, vectorized, W, unrolled, tiled, U, suffix, mask)
    end
end



