
@inline onefloat(::Type{T}) where {T} = one(sizeequivalentfloat(T))
@inline oneinteger(::Type{T}) where {T} = one(sizeequivalentint(T))
@inline zerofloat(::Type{T}) where {T} = zero(sizeequivalentfloat(T))
@inline zerointeger(::Type{T}) where {T} = zero(sizeequivalentint(T))


function lower_zero!(
    q::Expr, op::Operation, vectorized::Symbol, ls::LoopSet, u₁loop::Symbol, u₂loop::Symbol, U::Int, suffix::Union{Nothing,Int}, zerotyp::NumberType = zerotype(ls, op)
)
    mvar, opu₁, opu₂ = variable_name_and_unrolled(op, u₁loop, u₂loop, suffix)
    !isnothing(suffix) && !opu₂ && suffix > 0 && return
    typeT = ELTYPESYMBOL
    if zerotyp == HardInt
        newtypeT = gensym(:IntType)
        pushpreamble!(ls, Expr(:(=), newtypeT, Expr(:call, lv(:sizeequivalentint), ELTYPESYMBOL)))
        typeT = newtypeT
    elseif zerotyp == HardFloat
        newtypeT = gensym(:FloatType)
        pushpreamble!(ls, Expr(:(=), newtypeT, Expr(:call, lv(:sizeequivalentfloat), ELTYPESYMBOL)))
        typeT = newtypeT
    end
    if vectorized ∈ loopdependencies(op) || vectorized ∈ reducedchildren(op) || vectorized ∈ reduceddependencies(op)
        call = Expr(:call, lv(:vzero), VECTORWIDTHSYMBOL, typeT)
    else
        call = Expr(:call, :zero, typeT)
    end
    if opu₁
        broadcastsym = Symbol(mvar, "_#init#")
        pushpreamble!(ls, Expr(:(=), broadcastsym, call))
        # push!(q.args, Expr(:(=), broadcastsym, call))
        for u ∈ 0:U-1
            push!(q.args, Expr(:(=), Symbol(mvar, u), broadcastsym))
        end
    else
        push!(q.args, Expr(:(=), mvar, call))
    end
    nothing    
end
# Have to awkwardly search through `operations(ls)` to try and find op's child
function getparentsreductzero(ls::LoopSet, op::Operation)::Float64
    opname = name(op)
    for opp ∈ operations(ls)
        if name(opp) === opname && opp !== op && iscompute(opp) && search_tree(parents(opp), opname) && length(reduceddependencies(opp)) > 0
            return reduction_instruction_class(instruction(opp))
        end
    end
    throw("Reduct zero not found.")
end
function lower_constant!(
    q::Expr, op::Operation, vectorized::Symbol, ls::LoopSet, u₁loop::Symbol, u₂loop::Symbol, U::Int, suffix::Union{Nothing,Int}
)
    mvar, opu₁, opu₂ = variable_name_and_unrolled(op, u₁loop, u₂loop, suffix)
    !isnothing(suffix) && !opu₂ && suffix > 0 && return
    instruction = op.instruction
    constsym = instruction.instr
    reducedchildvectorized = vectorized ∈ reducedchildren(op)
    if reducedchildvectorized || vectorized ∈ loopdependencies(op)  || vectorized ∈ reduceddependencies(op)
        # call = Expr(:call, lv(:vbroadcast), W, Expr(:call, lv(:maybeconvert), typeT, constsym))
        call = if reducedchildvectorized && vectorized ∉ loopdependencies(op)
            instrclass = getparentsreductzero(ls, op)
            if instrclass == ADDITIVE_IN_REDUCTIONS
                Expr(:call, Expr(:(.), Expr(:(.), :LoopVectorization, QuoteNode(:SIMDPirates)), QuoteNode(:addscalar)), Expr(:call, lv(:vzero), VECTORWIDTHSYMBOL, ELTYPESYMBOL), constsym)
            elseif instrclass == MULTIPLICATIVE_IN_REDUCTIONS
                Expr(:call, Expr(:(.), Expr(:(.), :LoopVectorization, QuoteNode(:SIMDPirates)), QuoteNode(:mulscalar)), Expr(:call, lv(:vbroadcast), VECTORWIDTHSYMBOL, Expr(:call, :one, ELTYPESYMBOL)), constsym)
            else
                throw("Reductions of type $(reduction_zero(reinstrclass)) not yet supported; please file an issue as a reminder to take care of this.")
            end
        else
            Expr(:call, lv(:vbroadcast), VECTORWIDTHSYMBOL, constsym)
        end
        if opu₁
            broadcastsym = Symbol(mvar, "_#init#")
            push!(q.args, Expr(:(=), broadcastsym, call))
            for u ∈ 0:U-1
                push!(q.args, Expr(:(=), Symbol(mvar, u), broadcastsym))
            end
        else
            push!(q.args, Expr(:(=), mvar, call))
        end
    elseif opu₁
        for u ∈ 0:U-1
            push!(q.args, Expr(:(=), Symbol(mvar, u), constsym))
        end
    else
        push!(q.args, Expr(:(=), mvar, constsym))
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

# @inline maybeconvert(::Type{T}, s::Number) where {T} = convert(T, s)
# @inline maybeconvert(::Type{T}, s::T) where {T <: Number} = s
# @inline maybeconvert(::Type, s) = s


function lower_licm_constants!(ls::LoopSet)
    ops = operations(ls)
    for (id, sym) ∈ ls.preamble_symsym
        # setconstantop!(ls, ops[id], Expr(:call, lv(:maybeconvert), ls.T, sym))
        setconstantop!(ls, ops[id],  sym)
    end
    for (id,intval) ∈ ls.preamble_symint
        setop!(ls, ops[id], Expr(:call, lv(:sizeequivalentint), ELTYPESYMBOL, intval))
    end
    for (id,floatval) ∈ ls.preamble_symfloat
        setop!(ls, ops[id], Expr(:call, lv(:sizeequivalentfloat), ELTYPESYMBOL, floatval))
    end
    for (id,typ) ∈ ls.preamble_zeros
        instruction(ops[id]) === LOOPCONSTANT || continue
        if typ == IntOrFloat
            setconstantop!(ls, ops[id], Expr(:call, :zero, ELTYPESYMBOL))
        elseif typ == HardInt
            setconstantop!(ls, ops[id], Expr(:call, lv(:zerointeger), ELTYPESYMBOL))
        else#if typ == HardFloat
            setconstantop!(ls, ops[id], Expr(:call, lv(:zerofloat), ELTYPESYMBOL))
        end
    end
    for (id,typ) ∈ ls.preamble_ones
        if typ == IntOrFloat
            setop!(ls, ops[id], Expr(:call, :one, ELTYPESYMBOL))
        elseif typ == HardInt
            setop!(ls, ops[id], Expr(:call, lv(:oneinteger), ELTYPESYMBOL))
        else#if typ == HardFloat
            setop!(ls, ops[id], Expr(:call, lv(:onefloat), ELTYPESYMBOL))
        end
    end
end



