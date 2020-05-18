using VectorizationBase: vnoaliasstore!


@inline vstoreadditivereduce!(args...) = vnoaliasstore!(args...)
@inline vstoremultiplicativevereduce!(args...) = vnoaliasstore!(args...)
@inline function vstoreadditivereduce!(ptr::VectorizationBase.AbstractStridedPointer, v::VectorizationBase.SVec, i::NTuple{N,<:Integer}) where {N}
    vnoaliasstore!(ptr, SIMDPirates.vsum(v), i)
end
@inline function vstoreadditivereduce!(ptr::VectorizationBase.AbstractStridedPointer, v::VectorizationBase.SVec, i::NTuple{N,<:Integer}, m::VectorizationBase.Mask) where {N}
    vnoaliasstore!(ptr, SIMDPirates.vsum(v), i, m)
end
@inline function vstoremultiplicativevereduce!(ptr::VectorizationBase.AbstractStridedPointer, v::VectorizationBase.SVec, i::NTuple{N,<:Integer}) where {N}
    vnoaliasstore!(ptr, SIMDPirates.vprod(v), i)
end
@inline function vstoremultiplicativevereduce!(ptr::VectorizationBase.AbstractStridedPointer, v::VectorizationBase.SVec, i::NTuple{N,<:Integer}, m::VectorizationBase.Mask) where {N}
    vnoaliasstore!(ptr, SIMDPirates.vprod(v), i, m)
end

function storeinstr(op::Operation)
    opp = first(parents(op))
    if instruction(opp).instr === :identity
        opp = first(parents(opp))
    end
    defaultstoreop = :vnoaliasstore!
    # defaultstoreop = :vstore!
    instr = if iszero(length(reduceddependencies(opp)))
        defaultstoreop
    else
        instr_class = reduction_instruction_class(instruction(opp))
        if instr_class === ADDITIVE_IN_REDUCTIONS
            :vstoreadditivereduce!
        elseif instr_class === MULTIPLICATIVE_IN_REDUCTIONS
            :vstoremultiplicativevereduce!
        else #FIXME
            defaultstoreop
        end
    end
    lv(instr)
end

# const STOREOP = :vstore!
# variable_name(op::Operation, ::Nothing) = mangledvar(op)
# variable_name(op::Operation, suffix) = Symbol(mangledvar(op), suffix, :_)
# # variable_name(op::Operation, suffix, u::Int) = (n = variable_name(op, suffix); u < 0 ? n : Symbol(n, u))
function reduce_range!(q::Expr, toreduct::Symbol, instr::Instruction, Uh::Int, Uh2::Int)
    for u ∈ Uh:Uh2-1
        tru = Symbol(toreduct, u - Uh)
        push!(q.args, Expr(:(=), tru, Expr(instr, tru, Symbol(toreduct, u))))
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
        instr = Instruction(reduction_to_single_vector(op.instruction))
        reduce_range!(q, var, instr, Ulow, Uhigh)
    end
end

function reduce_expr!(q::Expr, toreduct::Symbol, instr::Instruction, U::Int)
    U == 1 && return nothing
    @assert U > 1 "U = $U somehow < 1"
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

# pvariable_name(op::Operation, ::Nothing) = mangledvar(first(parents(op)))
# pvariable_name(op::Operation, ::Nothing, ::Symbol) = mangledvar(first(parents(op)))
# pvariable_name(op::Operation, suffix) = Symbol(pvariable_name(op, nothing), suffix, :_)
# function pvariable_name(op::Operation, suffix, tiled::Symbol)
#     parent = first(parents(op))
#     mname = mangledvar(parent)
#     tiled ∈ loopdependencies(parent) ? Symbol(mname, suffix, :_) : mname
# end


function lower_conditionalstore_scalar!(
    q::Expr, op::Operation, ua::UnrollArgs, mask::Union{Nothing,Symbol,Unsigned}
)
    @unpack u₁, u₁loopsym, u₂loopsym, vectorized, suffix = ua
    mvar, opu₁, opu₂ = variable_name_and_unrolled(first(parents(op)), u₁loopsym, u₂loopsym, suffix)
    # var = pvariable_name(op, suffix, tiled)
    cond = last(parents(op))
    condvar, condu₁ = condvarname_and_unroll(cond, u₁loopsym, u₂loopsym, suffix, opu₂)
    loopdeps = loopdependencies(op)
    opu₁ = u₁loopsym ∈ loopdeps
    unrolled = opu₁ || u₂loopsym ∈ loopdeps
    ptr = unrolled ? offset_refname(op, ua) : refname(op)
    for u ∈ 0:u₁-1
        varname = varassignname(mvar, u, opu₁)
        condvarname = varassignname(condvar, u, condu₁)
        td = UnrollArgs(ua, u)
        push!(q.args, Expr(:&&, condvarname, Expr(:call, storeinstr(op), ptr, varname, mem_offset_u(op, td, unrolled))))
    end
    nothing
end
function lower_conditionalstore_vectorized!(
    q::Expr, op::Operation, ua::UnrollArgs, mask::Union{Nothing,Symbol,Unsigned}, isunrolled::Bool
)
    @unpack u₁, u₁loopsym, u₂loopsym, vectorized, suffix = ua
    loopdeps = loopdependencies(op)
    @assert vectorized ∈ loopdeps
    mvar, opu₁, opu₂ = variable_name_and_unrolled(first(parents(op)), u₁loopsym, u₂loopsym, suffix)
    # var = pvariable_name(op, suffix, tiled)
    if isunrolled
        umin = 0
        U = u₁
    else
        umin = -1
        U = 0
    end
    unrolled = opu₁ | opu₂
    ptr = unrolled ? offset_refname(op, ua) : refname(op)
    vecnotunrolled = vectorized !== u₁loopsym
    cond = last(parents(op))
    condvar, condu₁ = condvarname_and_unroll(cond, u₁loopsym, u₂loopsym, suffix, opu₂)
    # @show parents(op) cond condvar
    for u ∈ 0:U-1
        td = UnrollArgs(ua, u)
        name, mo = name_memoffset(mvar, op, td, opu₁, unrolled)
        condvarname = varassignname(condvar, u, condu₁)
        instrcall = Expr(:call, storeinstr(op), ptr, name, mo)
        if mask !== nothing && (vecnotunrolled || u == U - 1)
            push!(instrcall.args, Expr(:call, :&, condvarname, mask))
        else
            push!(instrcall.args, condvarname)
        end
        push!(q.args, instrcall)
    end
end

function lower_store_scalar!(
    q::Expr, op::Operation, ua::UnrollArgs, mask::Union{Nothing,Symbol,Unsigned}
)
    @unpack u₁, u₁loopsym, u₂loopsym, vectorized, suffix = ua
    mvar, opu₁, opu₂ = variable_name_and_unrolled(first(parents(op)), u₁loopsym, u₂loopsym, suffix)
    unrolled = opu₁ | opu₂
    ptr = unrolled ? offset_refname(op, ua) : refname(op)
    # var = pvariable_name(op, suffix, tiled)
    # parentisunrolled = unrolled ∈ loopdependencies(first(parents(op)))
    for u ∈ 0:u₁-1
        varname = varassignname(mvar, u, opu₁)
        td = UnrollArgs(ua, u)
        push!(q.args, Expr(:call, storeinstr(op), ptr, varname, mem_offset_u(op, td, unrolled)))
    end
    nothing
end
function lower_store_vectorized!(
    q::Expr, op::Operation, ua::UnrollArgs, mask::Union{Nothing,Symbol,Unsigned}, isunrolled::Bool
)
    @unpack u₁, u₁loopsym, u₂loopsym, vectorized, suffix = ua
    loopdeps = loopdependencies(op)
    @assert vectorized ∈ loopdeps
    mvar, opu₁, opu₂ = variable_name_and_unrolled(first(parents(op)), u₁loopsym, u₂loopsym, suffix)
    unrolled = opu₁ | opu₂
    ptr = unrolled ? offset_refname(op, ua) : refname(op)
    # var = pvariable_name(op, suffix, tiled)
    # parentisunrolled = unrolled ∈ loopdependencies(first(parents(op)))
    if isunrolled
        umin = 0
        U = u₁
    else
        umin = -1
        U = 0
    end
    vecnotunrolled = vectorized !== u₁loopsym
    for u ∈ umin:U-1
        td = UnrollArgs(ua, u)
        name, mo = name_memoffset(mvar, op, td, opu₁, unrolled)
        instrcall = Expr(:call, storeinstr(op), ptr, name, mo)
        if mask !== nothing && (vecnotunrolled || u == U - 1)
            push!(instrcall.args, mask)
        end
        push!(q.args, instrcall)
    end
end
function lower_store!(
    q::Expr, op::Operation, ua::UnrollArgs, mask::Union{Nothing,Symbol,Unsigned} = nothing
)
    @unpack u₁, u₁loopsym, u₂loopsym, vectorized, suffix = ua
    maybegesp_call!(q, op, ua)
    isunrolled = u₁loopsym ∈ loopdependencies(op)
    ua = UnrollArgs(ua, isunrolled ? u₁ : 1)
    if instruction(op).instr !== :conditionalstore!
        if vectorized ∈ loopdependencies(op)
            lower_store_vectorized!(q, op, ua, mask, isunrolled)
        else
            lower_store_scalar!(q, op, ua, mask)
        end
    else
        if vectorized ∈ loopdependencies(op)
            lower_conditionalstore_vectorized!(q, op, ua, mask, isunrolled)
        else
            lower_conditionalstore_scalar!(q, op, ua, mask)
        end
    end
end


