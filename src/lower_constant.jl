
@inline zerointeger(::Type{Float16}) = zero(Int16)
@inline zerointeger(::Type{Float32}) = zero(Int32)
@inline zerointeger(::Type{Float64}) = zero(Int64)
@inline zerointeger(::Type{I}) where {I<:Integer} = zero(I)
@inline zerofloat(::Type{Float16}) = zero(Float16)
@inline zerofloat(::Type{Float32}) = zero(Float32)
@inline zerofloat(::Type{Float64}) = zero(Float64)
@inline zerofloat(::Type{UInt16}) = zero(Float16)
@inline zerofloat(::Type{UInt32}) = zero(Float32)
@inline zerofloat(::Type{UInt64}) = zero(Float64)
@inline zerofloat(::Type{Int16}) = zero(Float16)
@inline zerofloat(::Type{Int32}) = zero(Float32)
@inline zerofloat(::Type{Int64}) = zero(Float64)


@inline oneinteger(::Type{Float16}) = one(Int16)
@inline oneinteger(::Type{Float32}) = one(Int32)
@inline oneinteger(::Type{Float64}) = one(Int64)
@inline oneinteger(::Type{I}) where {I<:Integer} = one(I)
@inline onefloat(::Type{Float16}) = one(Float16)
@inline onefloat(::Type{Float32}) = one(Float32)
@inline onefloat(::Type{Float64}) = one(Float64)
@inline onefloat(::Type{UInt16}) = one(Float16)
@inline onefloat(::Type{UInt32}) = one(Float32)
@inline onefloat(::Type{UInt64}) = one(Float64)
@inline onefloat(::Type{Int16}) = one(Float16)
@inline onefloat(::Type{Int32}) = one(Float32)
@inline onefloat(::Type{Int64}) = one(Float64)

@inline equivalentint(::Type{I}) where {I<:Integer} = I
@inline equivalentint(::Type{Float16}) = Int16
@inline equivalentint(::Type{Float32}) = Int32
@inline equivalentint(::Type{Float64}) = Int64
@inline equivalentfloat(::Type{Float16}) = Float16
@inline equivalentfloat(::Type{Float32}) = Float64
@inline equivalentfloat(::Type{Float64}) = Float64
@inline equivalentfloat(::Type{Int16}) = Float16
@inline equivalentfloat(::Type{Int32}) = Float64
@inline equivalentfloat(::Type{Int64}) = Float64
@inline equivalentfloat(::Type{UInt16}) = Float16
@inline equivalentfloat(::Type{UInt32}) = Float64
@inline equivalentfloat(::Type{UInt64}) = Float64

function lower_zero!(
    q::Expr, op::Operation, vectorized::Symbol, ls::LoopSet, unrolled::Symbol, U::Int, suffix::Union{Nothing,Int}, zerotyp::NumberType = zerotype(ls, op)
)
    W = ls.W; typeT = ls.T
    mvar = variable_name(op, suffix)
    if zerotyp == HardInt
        newtypeT = gensym(:IntType)
        pushpreamble!(ls, Expr(:(=), newtypeT, Expr(:call, lv(:equivalentint), typeT)))
        typeT = newtypeT
    elseif zerotyp == HardFloat
        newtypeT = gensym(:FloatType)
        pushpreamble!(ls, Expr(:(=), newtypeT, Expr(:call, lv(:equivalentfloat), typeT)))
        typeT = newtypeT
    end
    if vectorized ∈ loopdependencies(op) || vectorized ∈ reducedchildren(op) || vectorized ∈ reduceddependencies(op)
        call = Expr(:call, lv(:vzero), W, typeT)
    else
        call = Expr(:call, :zero, typeT)
    end
    if unrolled ∈ loopdependencies(op) || unrolled ∈ reducedchildren(op) || unrolled ∈ reduceddependencies(op)
        for u ∈ 0:U-1
            push!(q.args, Expr(:(=), Symbol(mvar, u), call))
        end
    else
        push!(q.args, Expr(:(=), mvar, call))
    end
    nothing    
end
function lower_constant!(
    q::Expr, op::Operation, vectorized::Symbol, ls::LoopSet, unrolled::Symbol, U::Int, suffix::Union{Nothing,Int}
)
    W = ls.W; typeT = ls.T
    instruction = op.instruction
    mvar = variable_name(op, suffix)
    constsym = instruction.instr
    if vectorized ∈ loopdependencies(op) || vectorized ∈ reducedchildren(op) || vectorized ∈ reduceddependencies(op)
        call = Expr(:call, lv(:vbroadcast), W, Expr(:call, lv(:maybeconvert), typeT, constsym))
        if unrolled ∈ loopdependencies(op) || unrolled ∈ reducedchildren(op) || unrolled ∈ reduceddependencies(op)
            for u ∈ 0:U-1
                push!(q.args, Expr(:(=), Symbol(mvar, u), call))
            end
        else
            push!(q.args, Expr(:(=), mvar, call))
        end
    else
        if unrolled ∈ loopdependencies(op) || unrolled ∈ reducedchildren(op) || unrolled ∈ reduceddependencies(op)
            for u ∈ 0:U-1
                push!(q.args, Expr(:(=), Symbol(mvar, u), constsym))
            end
        else
            push!(q.args, Expr(:(=), mvar, constsym))
        end
    end
    nothing
end


function setop!(ls, op, val)
    if instruction(op) === LOOPCONSTANT# && mangledvar(op) !== val
        pushpreamble!(ls, Expr(:(=), mangledvar(op), val))
    else
        pushpreamble!(ls, Expr(:(=), instruction(op).instr, val))
    end
    nothing
end
function setconstantop!(ls, op, val)
    if instruction(op) === LOOPCONSTANT# && mangledvar(op) !== val
        pushpreamble!(ls, Expr(:(=), mangledvar(op), val))
    end
    nothing
end

@inline maybeconvert(::Type{T}, s::Number) where {T} = convert(T, s)
@inline maybeconvert(::Type{T}, s::T) where {T <: Number} = s
@inline maybeconvert(::Type, s) = s


function lower_licm_constants!(ls::LoopSet)
    ops = operations(ls)
    for (id, sym) ∈ ls.preamble_symsym
        setconstantop!(ls, ops[id], Expr(:call, lv(:maybeconvert), ls.T, sym))
    end
    for (id,intval) ∈ ls.preamble_symint
        setop!(ls, ops[id], Expr(:call, lv(:sizeequivalentint), ls.T, intval))
    end
    for (id,floatval) ∈ ls.preamble_symfloat
        setop!(ls, ops[id], Expr(:call, lv(:sizeequivalentfloat), ls.T, floatval))
    end
    for (id,typ) ∈ ls.preamble_zeros
        instruction(ops[id]) === LOOPCONSTANT || continue
        if typ == IntOrFloat
            setconstantop!(ls, ops[id], Expr(:call, :zero, ls.T))
        elseif typ == HardInt
            setconstantop!(ls, ops[id], Expr(:call, lv(:zerointeger), ls.T))
        else#if typ == HardFloat
            setconstantop!(ls, ops[id], Expr(:call, lv(:zerofloat), ls.T))
        end
    end
    for (id,typ) ∈ ls.preamble_ones
        if typ == IntOrFloat
            setop!(ls, ops[id], Expr(:call, :one, ls.T))
        elseif typ == HardInt
            setop!(ls, ops[id], Expr(:call, lv(:oneinteger), ls.T))
        else#if typ == HardFloat
            setop!(ls, ops[id], Expr(:call, lv(:onefloat), ls.T))
        end
    end
end



