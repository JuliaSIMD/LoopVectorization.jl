variable_name(op::Operation, ::Nothing) = mangledvar(op)
variable_name(op::Operation, suffix) = Symbol(mangledvar(op), suffix, :_)
function reduce_range!(q::Expr, toreduct::Symbol, instr::Instruction, Uh::Int, Uh2::Int)
    for u ∈ 0:Uh-1
        tru = Symbol(toreduct, u)
        push!(q.args, Expr(:(=), tru, Expr(instr, tru, Symbol(toreduct, u + Uh))))
    end
    for u ∈ 2Uh:Uh2-1
        tru = Symbol(toreduct, u - 2Uh)
        push!(q.args, Expr(:(=), tru, Expr(instr, tru, Symbol(toreduct, u))))
    end
end
function reduce_range!(q::Expr, ls::LoopSet, Ulow::Int, Uhigh::Int)
    for or ∈ ls.outer_reductions
        op = ls.operations[or]
        var = mangledvar(op)
        temp = gensym(var)
        instr = Instruction(reduction_to_single_vector(op.instruction))
        reduce_range!(q, var, instr, Ulow, Uhigh)
    end
end

function reduce_expr!(q::Expr, toreduct::Symbol, instr::Instruction, U::Int)
    U == 1 && return nothing
    instr = Instruction(reduction_to_single_vector(instr))
    Uh2 = U
    iter = 0
    while true # combine vectors
        Uh = Uh2 >> 1
        reduce_range!(q, toreduct, instr, Uh, Uh2)
        Uh == 1 && break
        Uh2 = Uh
        iter += 1; iter > 4 && throw("Oops! This seems to be excessive unrolling.")
    end
    nothing
end

pvariable_name(op::Operation, ::Nothing) = mangledvar(first(parents(op)))
pvariable_name(op::Operation, suffix) = Symbol(pvariable_name(op, nothing), suffix, :_)

function lowered_variable_name(op::Operation, unrolled::Symbol, tiled::Symbol, u::Int, ::Nothing)
    varassignname(var, u, isunrolled)
end

function lower_conditionalstore_scalar!(
    q::Expr, op::Operation, vectorized::Symbol, W::Symbol, unrolled::Symbol, tiled::Symbol, U::Int,
    suffix::Union{Nothing,Int}, mask::Union{Nothing,Symbol,Unsigned}, isunrolled::Bool
)
    var = pvariable_name(op, suffix)
    cond = last(parents(op))
    condvar = if suffix === nothing || tiled ∉ loopdependencies(cond)
        pvariable_name(op, nothing)
    else
        pvariable_name(op, suffix)
    end
    condunrolled = unrolled ∈ loopdependencies(cond)
    ptr = refname(op)
    for u ∈ zero(Int32):Base.unsafe_trunc(Int32,U-1)
        varname = varassignname(var, u, isunrolled)
        condvarname = varassignname(condvar, u, condunrolled)
        td = UnrollArgs(u, unrolled, tiled, suffix)
        push!(q.args, Expr(:&&, condvarname, Expr(:call, lv(:store!), ptr, varname, mem_offset_u(op, td))))
    end
    nothing
end
function lower_conditionalstore_vectorized!(
    q::Expr, op::Operation, vectorized::Symbol, W::Symbol, unrolled::Symbol, tiled::Symbol, U::Int,
    suffix::Union{Nothing,Int}, mask::Union{Nothing,Symbol,Unsigned}, isunrolled::Bool
)
    loopdeps = loopdependencies(op)
    @assert unrolled ∈ loopdeps
    var = pvariable_name(op, suffix)
    if isunrolled
        umin = zero(Int32)
        U = U
    else
        umin = -one(Int32)
        U = 0
    end
    ptr = refname(op)
    vecnotunrolled = vectorized !== unrolled
    cond = last(parents(op))
    condvar = if suffix === nothing || tiled ∉ loopdependencies(cond)
        pvariable_name(op, nothing)
    else
        pvariable_name(op, suffix)
    end
    condunrolled = unrolled ∈ loopdependencies(cond)
    for u ∈ zero(Int32):Base.unsafe_trunc(Int32,U-1)
        td = UnrollArgs(u, unrolled, tiled, suffix)
        name, mo = name_memoffset(var, op, td, W, vecnotunrolled)
        condvarname = varassignname(condvar, u, condunrolled)
        instrcall = Expr(:call, lv(:vstore!), ptr, name, mo)
        if mask !== nothing && (vecnotunrolled || u == U - 1)
            push!(instrcall.args, Expr(:call, :&, condvarname, mask))
        else
            push!(instrcall.args, condvarname)
        end
        push!(q.args, instrcall)
    end
end

function lower_store_scalar!(
    q::Expr, op::Operation, vectorized::Symbol, W::Symbol, unrolled::Symbol, tiled::Symbol, U::Int,
    suffix::Union{Nothing,Int}, mask::Union{Nothing,Symbol,Unsigned}, isunrolled::Bool
)
    var = pvariable_name(op, suffix)
    ptr = refname(op)
    for u ∈ zero(Int32):Base.unsafe_trunc(Int32,U-1)
        varname = varassignname(var, u, isunrolled)
        td = UnrollArgs(u, unrolled, tiled, suffix)
        push!(q.args, Expr(:call, lv(:store!), ptr, varname, mem_offset_u(op, td)))
    end
    nothing
end
function lower_store_vectorized!(
    q::Expr, op::Operation, vectorized::Symbol, W::Symbol, unrolled::Symbol, tiled::Symbol, U::Int,
    suffix::Union{Nothing,Int}, mask::Union{Nothing,Symbol,Unsigned}, isunrolled::Bool
)
    loopdeps = loopdependencies(op)
    @assert unrolled ∈ loopdeps
    var = pvariable_name(op, suffix)
    if isunrolled
        umin = zero(Int32)
        U = U
    else
        umin = -one(Int32)
        U = 0
    end
    ptr = refname(op)
    vecnotunrolled = vectorized !== unrolled
    for u ∈ zero(Int32):Base.unsafe_trunc(Int32,U-1)
        td = UnrollArgs(u, unrolled, tiled, suffix)
        name, mo = name_memoffset(var, op, td, W, vecnotunrolled)
        instrcall = Expr(:call, lv(:vstore!), ptr, name, mo)
        if mask !== nothing && (vecnotunrolled || u == U - 1)
            push!(instrcall.args, mask)
        end
        push!(q.args, instrcall)
    end
end
function lower_store!(
    q::Expr, op::Operation, vectorized::Symbol, W::Symbol, unrolled::Symbol, tiled::Symbol, U::Int,
    suffix::Union{Nothing,Int}, mask::Union{Nothing,Symbol,Unsigned} = nothing
)
    # @show unrolled, tiled, U
    isunrolled = unrolled ∈ loopdependencies(op)
    U = isunrolled ? U : 1
    if instruction(op).instr !== :conditionalstore!
        if vectorized ∈ loopdependencies(op)
            lower_store_vectorized!(q, op, vectorized, W, unrolled, tiled, U, suffix, mask, isunrolled)
        else
            lower_store_scalar!(q, op, vectorized, W, unrolled, tiled, U, suffix, mask, isunrolled)
        end
    else
        if vectorized ∈ loopdependencies(op)
            lower_conditionalstore_vectorized!(q, op, vectorized, W, unrolled, tiled, U, suffix, mask, isunrolled)
        else
            lower_conditionalstore_scalar!(q, op, vectorized, W, unrolled, tiled, U, suffix, mask, isunrolled)
        end
    end
end


