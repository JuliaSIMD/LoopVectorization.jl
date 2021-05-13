function should_broadcast_op(op::Operation)
    (isvectorized(op) || iszero(length(children(op)))) && return false
    for opc ∈ children(op)
        (!isvectorized(op) || accesses_memory(op)) && return false
    end
    true
end


@inline sizeequivalentfloat(::Type{T}) where {T<:Union{Float16,Float32,Float64}} = T
@inline sizeequivalentfloat(::Type{T}) where {T <: Union{Int8,UInt8}} = Float32
@inline sizeequivalentfloat(::Type{T}) where {T <: Union{Int16,UInt16}} = Float16
@inline sizeequivalentfloat(::Type{T}) where {T <: Union{Int32,UInt32}} = Float32
@inline sizeequivalentfloat(::Type{T}) where {T <: Union{Int64,UInt64}} = Float64
@inline sizeequivalentint(::Type{T}) where {T <: Integer} = T
@inline sizeequivalentfloat(::Type{T}, x) where {T} = sizeequivalentfloat(T)(x)
@inline sizeequivalentint(::Type{T}, x) where {T} = sizeequivalentint(T)(x)

if (Sys.ARCH === :x86_64) || (Sys.ARCH === :i686)
    @inline widest_supported_integer(::True) = Int64
    @inline widest_supported_integer(::False) = Int32
    @inline sizeequivalentint(::Type{Float64}) = widest_supported_integer(VectorizationBase.has_feature(Val(:x86_64_avx512dq)))
else
    @inline sizeequivalentint(::Type{Float64}) = Int
end
@inline sizeequivalentint(::Type{Float32}) = Int32
@inline sizeequivalentint(::Type{Float16}) = Int16

@inline zerofloat(::Type{T}) where {T} = zero(sizeequivalentfloat(T))
@inline zerointeger(::Type{T}) where {T} = zero(sizeequivalentint(T))

function typeof_sym(ls::LoopSet, op::Operation, zerotyp::NumberType)
    if zerotyp == HardInt
        newtypeT = gensym(:IntType)
        pushpreamble!(ls, Expr(:(=), newtypeT, Expr(:call, lv(:sizeequivalentint), ELTYPESYMBOL)))
        newtypeT
    elseif zerotyp == HardFloat
        newtypeT = gensym(:FloatType)
        pushpreamble!(ls, Expr(:(=), newtypeT, Expr(:call, lv(:sizeequivalentfloat), ELTYPESYMBOL)))
        newtypeT
    else
        ELTYPESYMBOL
    end
end

function lower_zero!(
    q::Expr, op::Operation, ls::LoopSet, ua::UnrollArgs, zerotyp::NumberType = zerotype(ls, op)
)
  @unpack u₁, u₁loopsym, u₂loopsym, vloopsym, vloop, u₂max, suffix = ua
  mvar, opu₁, opu₂ = variable_name_and_unrolled(op, u₁loopsym, u₂loopsym, vloopsym, suffix, ls)
  !opu₂ && suffix > 0 && return
  # TODO: for u₁, needs to consider if reducedchildren are u₁-unrolled
  #       reductions need to consider reduct-status
  # if !opu₁
  #     opu₁ = u₁loopsym ∈ reducedchildren(op)
  # end
  typeT = typeof_sym(ls, op, zerotyp)
  # TODO: make should_broadcast_op handle everything.
  if isvectorized(op) || vloopsym ∈ reducedchildren(op) || vloopsym ∈ reduceddependencies(op) || should_broadcast_op(op)
    if opu₁ && u₁ > 1
      call = Expr(:call, lv(:zero_vecunroll), staticexpr(u₁), VECTORWIDTHSYMBOL, typeT, staticexpr(reg_size(ls)))
    else
      call = Expr(:call, lv(:_vzero), VECTORWIDTHSYMBOL, typeT, staticexpr(reg_size(ls)))
    end
  else
    call = Expr(:call, :zero, typeT)
    if opu₁ && u₁ > 1
      t = Expr(:tuple)
      for u ∈ 1:u₁
        push!(t.args, call)
      end
      call = Expr(:call, lv(:VecUnroll), t)
    end
  end
  if (suffix == -1) && opu₂
    for u ∈ 0:u₂max-1
      push!(q.args, Expr(:(=), Symbol(mvar, u, "__", Core.ifelse(opu₁, u₁, 1)), call))
    end
  else
    mvar = Symbol(mvar, '_', Core.ifelse(opu₁, u₁, 1))
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
    throw("Reduct zero not found for operation $(name(op)).")
end
vecbasefunc(f) = Expr(:(.), Expr(:(.), :LoopVectorization, QuoteNode(:VectorizationBase)), QuoteNode(f))
function lower_constant!(
    q::Expr, op::Operation, ls::LoopSet, ua::UnrollArgs
)
  @unpack u₁, u₁loopsym, u₂loopsym, vloopsym, u₂max, suffix = ua
  mvar, opu₁, opu₂ = variable_name_and_unrolled(op, u₁loopsym, u₂loopsym, vloopsym, suffix, ls)
  !opu₂ && suffix > 0 && return
  instr = instruction(op)
  instr.mod === GLOBALCONSTANT && return
  constsym = constantopname(op)# instr.instr
  reducedchildvectorized = vloopsym ∈ reducedchildren(op)
  if reducedchildvectorized || isvectorized(op) || vloopsym ∈ reduceddependencies(op) || should_broadcast_op(op)
    # call = Expr(:call, lv(:vbroadcast), W, Expr(:call, lv(:maybeconvert), typeT, constsym))
    call = if reducedchildvectorized && vloopsym ∉ loopdependencies(op)
      instrclass = getparentsreductzero(ls, op)
      if instrclass == ADDITIVE_IN_REDUCTIONS
        Expr(:call, vecbasefunc(:addscalar), Expr(:call, lv(:vzero), VECTORWIDTHSYMBOL, ELTYPESYMBOL), constsym)
      elseif instrclass == MULTIPLICATIVE_IN_REDUCTIONS
        Expr(:call, vecbasefunc(:mulscalar), Expr(:call, lv(:vbroadcast), VECTORWIDTHSYMBOL, Expr(:call, :one, ELTYPESYMBOL)), constsym)
      elseif instrclass == MAX
        Expr(:call, vecbasefunc(:maxscalar), Expr(:call, lv(:vbroadcast), VECTORWIDTHSYMBOL, Expr(:call, :typemin, ELTYPESYMBOL)), constsym)
      elseif instrclass == MIN
        Expr(:call, vecbasefunc(:minscalar), Expr(:call, lv(:vbroadcast), VECTORWIDTHSYMBOL, Expr(:call, :typemax, ELTYPESYMBOL)), constsym)
      else
        throw("Reductions of type $(reduction_zero(reinstrclass)) not yet supported; please file an issue as a reminder to take care of this.")
      end
    else
      Expr(:call, lv(:vbroadcast), VECTORWIDTHSYMBOL, constsym)
    end
    if opu₁ && u₁ > 1
      # broadcastsym = Symbol(mvar, "_#init#")
      # push!(q.args, Expr(:(=), broadcastsym, call))
      t = Expr(:tuple)
      for u ∈ 1:u₁
        push!(t.args, call)
      end
      call = Expr(:call, lv(:VecUnroll), t)
    end
  elseif opu₁ && u₁ > 1
    t = Expr(:tuple)
    for u ∈ 1:u₁
      push!(t.args, constsym)
    end
    call = Expr(:call, lv(:VecUnroll), t)
  elseif opu₂ & (suffix == -1)
    for u ∈ 0:u₂max-1
      push!(q.args, Expr(:(=), Symbol(mvar, u, "__", 1), constsym))
    end
    return nothing
  else
    push!(q.args, Expr(:(=), Symbol(mvar, '_', 1), constsym))
    return nothing
  end
  u₁tag = Core.ifelse(opu₁, u₁, 1)
  if opu₂ & (suffix == -1)
    for u ∈ 0:u₂max-1
      push!(q.args, Expr(:(=), Symbol(mvar, u, "__", u₁tag), call))
    end
  else
    mvar = Symbol(mvar, '_', u₁tag)
    push!(q.args, Expr(:(=), mvar, call))
  end
  nothing
end

isconstantop(op::Operation) = (instruction(op) == LOOPCONSTANT) || (isconstant(op) && length(loopdependencies(op)) == 0)
function constantopname(op::Operation)
  instr = instruction(op)
  if instr === LOOPCONSTANT
    Symbol(mangledvar(op), '_', 1)
  else
    instr.instr
  end
end
function setop!(ls, op, val)
  pushpreamble!(ls, Expr(:(=), constantopname(op), val))
  nothing
end
function setconstantop!(ls, op, val)
    if instruction(op) === LOOPCONSTANT# && mangledvar(op) !== val
        pushpreamble!(ls, Expr(:(=), Symbol(mangledvar(op), '_', 1), val))
        # pushpreamble!(ls, Expr(:(=), mangledvar(op), val))
    end
    nothing
end

# @inline maybeconvert(::Type{T}, s::Number) where {T} = convert(T, s)
# @inline maybeconvert(::Type{T}, s::T) where {T <: Number} = s
# @inline maybeconvert(::Type, s) = s


function lower_licm_constants!(ls::LoopSet)
    ops = operations(ls)
    for (id, sym) ∈ ls.preamble_symsym
        isouterreduct = false
        for or ∈ ls.outer_reductions
            isouterreduct |= mangledvar(ls.operations[or]) === mangledvar(ops[id])
        end
        isouterreduct || setconstantop!(ls, ops[id],  sym)
        # setconstantop!(ls, ops[id],  sym)
        # setconstantop!(ls, ops[id], Expr(:call, lv(:maybeconvert), ls.T, sym))
    end
    for (id,(intval,intsz,signed)) ∈ ls.preamble_symint
        if intsz == 1
            setop!(ls, ops[id], intval % Bool)
        elseif signed
            setop!(ls, ops[id], Expr(:call, lv(:sizeequivalentint), ELTYPESYMBOL, intval))
        else
            setop!(ls, ops[id], Expr(:call, lv(:sizeequivalentint), ELTYPESYMBOL, intval % UInt))
        end
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
    for (id,f) ∈ ls.preamble_funcofeltypes
        setop!(ls, ops[id], Expr(:call, reduction_zero(f), ELTYPESYMBOL))
    end
end
