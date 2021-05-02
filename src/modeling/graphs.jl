struct MaybeKnown
    hint::Int
    sym::Symbol
    known::Bool
end
struct Loop
    itersymbol::Symbol
    start::MaybeKnown
    stop::MaybeKnown
    step::MaybeKnown
    rangesym::Symbol# === Symbol("") means loop is static
    lensym::Symbol
end

struct UnrollSymbols
    u₁loopsym::Symbol
    u₂loopsym::Symbol
    vloopsym::Symbol
end
struct UnrollArgs
    u₁loop::Loop
    u₂loop::Loop
    vloop::Loop
    u₁::Int
    u₂max::Int
    suffix::Int # -1 means not tiled
end
UnPack.unpack(ua::UnrollArgs, ::Val{:u₁loopsym}) = getfield(getfield(ua, :u₁loop), :itersymbol)
UnPack.unpack(ua::UnrollArgs, ::Val{:u₂loopsym}) = getfield(getfield(ua, :u₂loop), :itersymbol)
UnPack.unpack(ua::UnrollArgs, ::Val{:vloopsym}) = getfield(getfield(ua, :vloop), :itersymbol)
UnPack.unpack(ua::UnrollArgs, ::Val{:u₁step}) = getfield(getfield(ua, :u₁loop), :step)
UnPack.unpack(ua::UnrollArgs, ::Val{:u₂step}) = getfield(getfield(ua, :u₂loop), :step)
UnPack.unpack(ua::UnrollArgs, ::Val{:vstep}) = getfield(getfield(ua, :vloop), :step)


function UnrollArgs(ua::UnrollArgs, u₁::Int)
    @unpack u₁loop, u₂loop, vloop, u₂max, suffix = ua
    UnrollArgs(u₁loop, u₂loop, vloop, u₁, u₂max, suffix)
end
# UnrollSymbols(ua::UnrollArgs) = UnrollSymbols(ua.u₁loopsym, ua.u₂loopsym, ua.vloopsym)

# isfirst(ua::UnrollArgs{Nothing}) = iszero(ua.u₁)
# isfirst(ua::UnrollArgs{Int}) = iszero(ua.u₁) & iszero(ua.suffix)

struct UnrollSpecification
    u₁loopnum::Int
    u₂loopnum::Int
    vloopnum::Int
    u₁::Int
    u₂::Int
end
# UnrollSpecification(ls::LoopSet, u₁loop::Loop, vloopsym::Symbol, u₁, u₂) = UnrollSpecification(ls, u₁loop.itersymbol, vloopsym, u₁, u₂)
function UnrollSpecification(us::UnrollSpecification, u₁, u₂)
    @unpack u₁loopnum, u₂loopnum, vloopnum = us
    UnrollSpecification(u₁loopnum, u₂loopnum, vloopnum, u₁, u₂)
end
# function UnrollSpecification(us::UnrollSpecification; u₁ = us.u₁, u₂ = us.u₂)
#     @unpack u₁loopnum, u₂loopnum, vloopnum = us
#     UnrollSpecification(u₁loopnum, u₂loopnum, vloopnum, u₁, u₂)
# end
isunrolled1(us::UnrollSpecification, n::Int) = us.u₁loopnum == n
isunrolled2(us::UnrollSpecification, n::Int) = !isunrolled1(us, n) && us.u₂loopnum == n
isvectorized(us::UnrollSpecification, n::Int) = us.vloopnum == n
function unrollfactor(us::UnrollSpecification, n::Int)
    @unpack u₁loopnum, u₂loopnum, u₁, u₂ = us
    (u₁loopnum == n) ? u₁ : ((u₂loopnum == n) ? u₂ : 1)
end
function pushexpr!(ex::Expr, mk::MaybeKnown)
    if isknown(mk)
        push!(ex.args, staticexpr(gethint(mk)))
    else
        push!(ex.args, getsym(mk))
    end
    nothing
end
pushexpr!(ex::Expr, x::Union{Symbol,Expr}) = (push!(ex.args, x); nothing)
pushexpr!(ex::Expr, x::Integer) = (push!(ex.args, staticexpr(convert(Int, x))); nothing)
MaybeKnown(x::Integer) = MaybeKnown(convert(Int, x), Symbol("##UNDEFINED##"), true)
MaybeKnown(x::Integer, default::Int) = MaybeKnown(x)
MaybeKnown(x::Symbol, default::Int) = MaybeKnown(default, x, false)

isknown(mk::MaybeKnown) = getfield(mk, :known)
getsym(mk::MaybeKnown) = getfield(mk, :sym)
gethint(mk::MaybeKnown) = getfield(mk, :hint)
Base.isone(mk::MaybeKnown) = isknown(mk) && isone(gethint(mk))
Base.iszero(mk::MaybeKnown) = isknown(mk) && iszero(gethint(mk))

function Loop(
    itersymbol::Symbol, start::Union{Int,Symbol}, stop::Union{Int,Symbol}, step::Union{Int,Symbol},
    rangename::Symbol, lensym::Symbol
)
    Loop(itersymbol, MaybeKnown(start, 1), MaybeKnown(stop, 1024), MaybeKnown(step, 1), rangename, lensym)
end
startstopΔ(loop::Loop) = gethint(last(loop)) - gethint(first(loop))
function Base.length(loop::Loop)
    l = startstopΔ(loop)
    s = gethint(step(loop))
    (isone(s) ? l : cld(l, s)) + 1
end
Base.first(l::Loop) = getfield(l, :start)
Base.last(l::Loop) = getfield(l, :stop)
Base.step(l::Loop) = getfield(l, :step)

isstaticloop(l::Loop) = isknown(first(l)) & isknown(last(l)) & isknown(step(l))
unitstep(l::Loop) = isone(step(l))



function startloop(loop::Loop, itersymbol)
    start = first(loop)
    if isknown(start)
        Expr(:(=), itersymbol, gethint(start))
    else
        Expr(:(=), itersymbol, Expr(:call, lv(:Int), getsym(start)))
    end
end
# mulexpr(a,b) = Expr(:call, lv(:vmul_fast), a, b)

pushmulexpr!(q, a, b) = (push!(q.args, mulexpr(a, b)); nothing)
function pushmulexpr!(q, a, b::Integer)
    if isone(b)
        push!(q.args, a)
    else
        push!(q.args, mulexpr(a, b))
    end
    nothing
end

# function arithmetic_expr(f, a, b)
#     call = Expr(:call, lv(f))
#     if isa(a, MaybeKnown)
#         pushexpr!(
# end
isknown(x::Union{Symbol,Expr}) = false
isknown(x::Integer) = true
gethint(a::Integer) = a
addexpr(a,b) = arithmeticexpr(+, :vadd_fast, a, b)
subexpr(a,b) = arithmeticexpr(-, :vsub_fast, a, b)
mulexpr(a,b) = arithmeticexpr(*, :vmul_fast, a, b)
lazymulexpr(a,b) = arithmeticexpr(*, :lazymul, a, b)
function arithmeticexpr(op, f, a, b)
    if isknown(a) & isknown(b)
        return staticexpr(op(gethint(a), gethint(b)))
    else
        ex = Expr(:call, lv(f))
        pushexpr!(ex, a)
        pushexpr!(ex, b)
        return ex
    end
end
mulexpr(a,b,c) = arithmeticexpr(*, 1, :vmul_fast, a, b, c)
addexpr(a,b,c) = arithmeticexpr(+, 0, :vadd_fast, a, b, c)
function arithmeticexpr(op, init, f, a, b, c)
    ex = Expr(:call, lv(f))
    p = init
    if isknown(a)
        p = op(p, gethint(a))
        known = 1
    else
        pushexpr!(ex, a)
        known = 0
    end
    if isknown(b)
        p = op(p, gethint(b))
        known += 1
    else
        pushexpr!(ex, b)
    end
    if isknown(c)
        p = op(p, gethint(c))
        known += 1
    elseif known == 0
        ex = Expr(:call, lv(f), ex)
        pushexpr!(ex, c)
    end
    if known == 3
        return staticexpr(p)
    else
        if known == 2
            pushexpr!(ex, p)
            return ex
        elseif known == 1
            return Expr(:call, lv(f), ex, staticexpr(p))
        else#known == 0
            return ex
        end
    end
end

function addexpr(ex, incr::Integer)
  if incr > 0
    f = :vadd_fast
  else
    f = :vsub_fast
    incr = -incr
  end
  expr = Expr(:call, lv(f))
  pushexpr!(expr, ex)
  pushexpr!(expr, convert(Int, incr))
  expr
end
staticmulincr(ptr, incr) = Expr(:call, lv(:staticmul), Expr(:call, :eltype, ptr), incr)

@inline cmpend(i::Int, r::CloseOpen) = i < getfield(r,:upper)
@inline cmpend(i::Int, r::AbstractUnitRange) = i ≤ last(r)
@inline cmpend(i::Int, r::AbstractRange) = i ≤ last(r)
# @inline cmpend(i::Int, r::AbstractRange) = @show i last(r) i ≤ last(r)
# @inline cmpend(i::Int, r::AbstractRange) = i ≤ vsub_fast(last(r), step(r))

@inline vcmpend(i::Int, r::CloseOpen, ::StaticInt{W}) where {W} = i ≤ vsub_fast(getfield(r,:upper), W)
@inline vcmpend(i::Int, r::AbstractUnitRange, ::StaticInt{W}) where {W} = i ≤ vsub_fast(last(r), W-1)
# i = 0
# i += 4*3 # i = 12
@inline vcmpend(i::Int, r::AbstractRange, ::StaticInt{W}) where {W} = i ≤ vsub_fast(last(r), vsub_fast(W*step(r), 1))
# @inline vcmpend(i::Int, r::AbstractRange, ::StaticInt{W}) where {W} = i ≤ vsub_fast(last(r), W*step(r))
# @inline vcmpend(i::Int, r::AbstractRange, ::StaticInt{W}) where {W} = @show i m = vsub_fast(last(r), W*step(r)) i ≤ m
# @inline vcmpend(i::Int, r::AbstractRange, ::StaticInt{W}) where {W} = i ≤ vsub_fast(last(r), W)

function staticloopexpr(loop::Loop)
  f = first(loop)
  s = step(loop)
  l = last(loop)
  if isone(s)
    Expr(:call, GlobalRef(Base, :(:)), staticexpr(gethint(f)), staticexpr(gethint(l)))
  else
    Expr(:call, GlobalRef(Base, :(:)), staticexpr(gethint(f)), staticexpr(gethint(s)), staticexpr(gethint(l)))
  end
end
function vec_looprange(loop::Loop, UF::Int, mangledname)
  if loop.rangesym === Symbol("") # means loop is static
    vec_looprange(UF, mangledname, staticloopexpr(loop))
  else
    vec_looprange(UF, mangledname, loop.rangesym)
  end
end
function vec_looprange(UF::Int, mangledname, r::Union{Expr,Symbol})
  if isone(UF)
    Expr(:call, lv(:vcmpend), mangledname, r, VECTORWIDTHSYMBOL)
  else
    Expr(:call, lv(:vcmpend), mangledname, r, mulexpr(VECTORWIDTHSYMBOL, UF))
  end
end
function looprange(loop::Loop, UF::Int, mangledname)
  if loop.rangesym === Symbol("") # means loop is static
    looprange(UF, mangledname, staticloopexpr(loop))
  else
    looprange(UF, mangledname, loop.rangesym)
  end
end
function looprange(UF::Int, mangledname, r::Union{Expr,Symbol})
  if isone(UF)
    Expr(:call, lv(:cmpend), mangledname, r)
  else
    Expr(:call, lv(:vcmpend), mangledname, r, staticexpr(UF))
  end
end

function terminatecondition(
    loop::Loop, us::UnrollSpecification, n::Int, mangledname::Symbol, inclmask::Bool, UF::Int = unrollfactor(us, n)
)
    if !isvectorized(us, n)
        looprange(loop, UF, mangledname)
    elseif inclmask
        looprange(loop, 1, mangledname)
    else
        vec_looprange(loop, UF, mangledname) # may not be u₂loop
    end
end

function incrementloopcounter(us::UnrollSpecification, n::Int, mangledname::Symbol, UF::Int, l::Loop)
    incr = step(l)
    if isknown(incr)
        incrementloopcounter(us, n, mangledname, UF * gethint(incr))
    else
        incrementloopcounter(us, n, mangledname, UF, getsym(incr))
    end
end
function incrementloopcounter(us::UnrollSpecification, n::Int, mangledname::Symbol, UF::Int)
    if isvectorized(us, n)
        if isone(UF)
            Expr(:(=), mangledname, addexpr(VECTORWIDTHSYMBOL, mangledname))
        else
            Expr(:(=), mangledname, addexpr(mulexpr(VECTORWIDTHSYMBOL, staticexpr(UF)), mangledname))
        end
    else
        Expr(:(=), mangledname, addexpr(mangledname, UF))
    end
end
function incrementloopcounter(us::UnrollSpecification, n::Int, mangledname::Symbol, UF::Int, incr::Symbol)
    if isvectorized(us, n)
        if isone(UF)
            Expr(:(=), mangledname, addexpr(mulexpr(VECTORWIDTHSYMBOL, incr), mangledname))
        else
            Expr(:(=), mangledname, addexpr(mulexpr(mulexpr(VECTORWIDTHSYMBOL, staticexpr(UF)), incr), mangledname))
        end
    else
        Expr(:(=), mangledname, addexpr(mangledname, mulexpr(incr, UF)))
    end
end

function incrementloopcounter!(q, us::UnrollSpecification, n::Int, UF::Int, l::Loop)
    incr = step(l)
    if isknown(incr)
        incrementloopcounter!(q, us, n, UF * gethint(incr))
    else
        incrementloopcounter!(q, us, n, UF, getsym(incr))
    end
end
function incrementloopcounter!(q, us::UnrollSpecification, n::Int, UF::Int)
    if isvectorized(us, n)
        if isone(UF)
            push!(q.args, VECTORWIDTHSYMBOL)
        else
            push!(q.args, mulexpr(VECTORWIDTHSYMBOL, staticexpr(UF)))
        end
    else
        push!(q.args, staticexpr(UF))
    end
end
function incrementloopcounter!(q, us::UnrollSpecification, n::Int, UF::Int, incr::Symbol)
    if isvectorized(us, n)
        if isone(UF)
            push!(q.args, mulexpr(VECTORWIDTHSYMBOL, incr))
        else
            push!(q.args, mulexpr(mulexpr(VECTORWIDTHSYMBOL, staticexpr(UF)), incr))
        end
    else
        push!(q.args, mulexpr(staticexpr(UF), incr))
    end
end
# function looplengthexpr(loop::Loop)
#     if loop.stopexact
#         if loop.startexact
#             return length(loop)
#         # elseif loop.rangename === Symbol("")
#             # return Expr(:call, lv(:vsub_fast), loop.stophint + 1, loop.startsym)
#         end
#     elseif loop.startexact
#         if isone(loop.starthint)
#             return loop.stopsym
#         # elseif loop.rangename === Symbol("")
#             # return Expr(:call, lv(:vsub_fast), loop.stopsym, loop.starthint - 1)
#         end
#     # elseif loop.rangename === Symbol("")
#         # return Expr(:call, lv(:vsub_fast), loop.stopsym, Expr(:call, lv(:staticm1), loop.startsym))
#     end
#     Expr(:call, lv(:static_length), loop.rangename)
# end
# use_expect() = false
# use_expect() = true
# function looplengthexpr(loop, n)
#     le = looplengthexpr(loop)
#     # if false && use_expect() && isone(n) && !isstaticloop(loop)
#     #     le = expect(le)
#     #     push!(le.args, Expr(:call, Expr(:curly, :Val, length(loop))))
#     # end
#     le
# end

# load/compute/store × isunrolled × istiled × pre/post loop × Loop number
struct LoopOrder <: AbstractArray{Vector{Operation},5}
    oporder::Vector{Vector{Operation}}
    loopnames::Vector{Symbol}
    bestorder::Vector{Symbol}
end
# function LoopOrder(N::Int)
#     LoopOrder(
#         [ Operation[] for _ ∈ 1:8N ],
#         Vector{Symbol}(undef, N), Vector{Symbol}(undef, N)
#     )
# end
LoopOrder() = LoopOrder(Vector{Operation}[],Symbol[],Symbol[])
Base.empty!(lo::LoopOrder) = foreach(empty!, lo.oporder)
function Base.resize!(lo::LoopOrder, N::Int)
    Nold = length(lo.loopnames)
    resize!(lo.oporder, 8N)
    for n ∈ 8Nold+1:8N
        lo.oporder[n] = Operation[]
    end
    resize!(lo.loopnames, N)
    resize!(lo.bestorder, N)
    lo
end
Base.size(lo::LoopOrder) = (2,2,2,length(lo.loopnames))
Base.@propagate_inbounds Base.getindex(lo::LoopOrder, i::Int) = lo.oporder[i]
Base.@propagate_inbounds Base.getindex(lo::LoopOrder, i::Vararg{Int,K}) where {K} = lo.oporder[LinearIndices(size(lo))[i...]]

@enum NumberType::Int8 HardInt HardFloat IntOrFloat INVALID


struct LoopStartStopManager
    terminators::Vector{Int}
    incrementedptrs::Vector{Vector{ArrayReferenceMeta}}
    uniquearrayrefs::Vector{ArrayReferenceMeta}
end
# Must make it easy to iterate
# outer_reductions is a vector of indices (within operation vectors) of the reduction operation, eg the vmuladd op in a dot product
# O(N) search is faster at small sizes
mutable struct LoopSet
    loopsymbols::Vector{Symbol}
    loopsymbol_offsets::Vector{Int}  # symbol loopsymbols[i] corresponds to loops[lso[i]+1:lso[i+1]] (CartesianIndex handling)
    loops::Vector{Loop}
    opdict::Dict{Symbol,Operation}
    operations::Vector{Operation} # Split them to make it easier to iterate over just a subset
    operation_offsets::Vector{Int}
    outer_reductions::Vector{Int} # IDs of reduction operations that need to be reduced at end.
    loop_order::LoopOrder
    preamble::Expr
    prepreamble::Expr # performs extractions that must be performed first, and don't need further registering
    preamble_symsym::Vector{Tuple{Int,Symbol}}
    preamble_symint::Vector{Tuple{Int,Tuple{Int,Int32,Bool}}}
    preamble_symfloat::Vector{Tuple{Int,Float64}}
    preamble_zeros::Vector{Tuple{Int,NumberType}}
    preamble_funcofeltypes::Vector{Tuple{Int,Float64}}
    includedarrays::Vector{Symbol}
    includedactualarrays::Vector{Symbol}
    syms_aliasing_refs::Vector{Symbol}
    refs_aliasing_syms::Vector{ArrayReferenceMeta}
    cost_vec::Matrix{Float64}
    reg_pres::Matrix{Float64}
    included_vars::Vector{Bool}
    place_after_loop::Vector{Bool}
    unrollspecification::UnrollSpecification
    loadelimination::Bool
    lssm::LoopStartStopManager
    vector_width::Int
    symcounter::Int
    isbroadcast::Bool
    register_size::Int
    register_count::Int
    cache_linesize::Int
    cache_size::Tuple{Int,Int,Int}
    ureduct::Int
    equalarraydims::Vector{Tuple{Vector{Symbol},Vector{Int}}}
    omop::OffsetLoadCollection
    loopordermap::Vector{Int}
    loopindexesbit::Vector{Bool}
    mod::Symbol
    LoopSet() = new()
end

function UnrollArgs(ls::LoopSet, u₁::Int, unrollsyms::UnrollSymbols, u₂max::Int, suffix::Int)
    @unpack u₁loopsym, u₂loopsym, vloopsym = unrollsyms
    u₁loop = getloop(ls, u₁loopsym)
    u₂loop = u₂loopsym === Symbol("##undefined##") ? u₁loop : getloop(ls, u₂loopsym)
    vloop = getloop(ls, vloopsym)
    UnrollArgs(u₁loop, u₂loop, vloop, u₁, u₂max, suffix)
end


function cost_vec_buf(ls::LoopSet)
    cv = @view(ls.cost_vec[:,2])
    @inbounds for i ∈ 1:4
        cv[i] = 0.0
    end
    cv
end
function reg_pres_buf(ls::LoopSet)
    ps = @view(ls.reg_pres[:,2])
    @inbounds for i ∈ 1:4
        ps[i] = 0
    end
    ps[4] = reg_count(ls)
    ps
end
function save_tilecost!(ls::LoopSet)
    @inbounds for i ∈ 1:4
        ls.cost_vec[i,1] = ls.cost_vec[i,2]
        ls.reg_pres[i,1] = ls.reg_pres[i,2]
    end
    # ls.reg_pres[5,1] = ls.reg_pres[5,2]
end
function set_hw!(ls::LoopSet, rs::Int, rc::Int, cls::Int, l1::Int, l2::Int, l3::Int)
    ls.register_size = rs
    ls.register_count = rc
    ls.cache_linesize = cls
    ls.cache_size = (l1,l2,l3)
    # ls.opmask_register[] = omr
    nothing
end
available_registers() = ifelse(has_opmask_registers(), register_count(), register_count() - One())
function set_hw!(ls::LoopSet)
    set_hw!(
        ls, Int(register_size()), Int(available_registers()), Int(cache_linesize()),
        Int(cache_size(StaticInt(1))), Int(cache_size(StaticInt(2))), Int(cache_size(StaticInt(3)))
    )
end
reg_size(ls::LoopSet) = ls.register_size
reg_count(ls::LoopSet) = ls.register_count
cache_lnsze(ls::LoopSet) = ls.cache_linesize
cache_sze(ls::LoopSet) = ls.cache_size

pushprepreamble!(ls::LoopSet, ex) = push!(ls.prepreamble.args, ex)
function pushpreamble!(ls::LoopSet, op::Operation, v::Symbol)
    if v !== mangledvar(op)
        push!(ls.preamble_symsym, (identifier(op),v))
    end
    nothing
end

function integer_description(@nospecialize(v::Integer))::Tuple{Int,Int32,Bool}
    if v isa Bool
        ((v % Int)::Int, one(Int32), false)
    else
        ((v % Int)::Int, ((8sizeof(v))%Int32)::Int32, (v isa Signed)::Bool)
    end
end

function pushpreamble!(ls::LoopSet, op::Operation, v::Number)
    typ = v isa Integer ? HardInt : HardFloat
    id = identifier(op)
    if iszero(v)
        push!(ls.preamble_zeros, (id, typ))
    elseif isone(v)
        push!(ls.preamble_funcofeltypes, (id, MULTIPLICATIVE_IN_REDUCTIONS))
    elseif v isa Integer
        push!(ls.preamble_symint, (id, integer_description(v)))
    else
        push!(ls.preamble_symfloat, (id, convert(Float64,v)))
    end
end
pushpreamble!(ls::LoopSet, ex::Expr) = push!(ls.preamble.args, ex)
# function pushpreamble!(ls::LoopSet, op::Operation, RHS::Expr)
#     c = gensym(:licmconst)
#     if RHS.head === :call && first(RHS.args) === :zero
#         push!(ls.preamble_zeros, (identifier(op), IntOrFloat))
#     elseif RHS.head === :call && first(RHS.args) === :one
#         push!(ls.preamble_funcofeltypes, (identifier(op), MULTIPLICATIVE_IN_REDUCTIONS))
#     else
#         pushpreamble!(ls, Expr(:(=), c, RHS))
#         pushpreamble!(ls, op, c)
#     end
#     nothing
# end
function zerotype(ls::LoopSet, op::Operation)
    opid = identifier(op)
    for (id,typ) ∈ ls.preamble_zeros
        id == opid && return typ
    end
    INVALID
end

includesarray(ls::LoopSet, array::Symbol) = array ∈ ls.includedarrays

function LoopSet(mod::Symbol)
    ls = LoopSet()
    ls.loopsymbols = Symbol[]
    ls.loopsymbol_offsets =  [0]
    ls.loops = Loop[]
    ls.opdict = Dict{Symbol,Operation}()
    ls.operations = Operation[]
    ls.operation_offsets = Int[0]
    ls.outer_reductions = Int[]
    ls.loop_order = LoopOrder()
    ls.preamble = Expr(:block)
    ls.prepreamble = Expr(:block)
    ls.preamble_symsym = Tuple{Int,Symbol}[]
    ls.preamble_symint = Tuple{Int,Tuple{Int,Int32,Bool}}[]
    ls.preamble_symfloat = Tuple{Int,Float64}[]
    ls.preamble_zeros = Tuple{Int,NumberType}[]
    ls.preamble_funcofeltypes = Tuple{Int,Float64}[]
    ls.includedarrays = Symbol[]
    ls.includedactualarrays = Symbol[]
    ls.syms_aliasing_refs = Symbol[]
    ls.refs_aliasing_syms = ArrayReferenceMeta[]
    ls.cost_vec = Matrix{Float64}(undef, 4, 2)
    ls.reg_pres = Matrix{Float64}(undef, 4, 2)
    ls.included_vars = Bool[]
    ls.place_after_loop = Bool[]
    ls.unrollspecification
    ls.loadelimination = false
    ls.vector_width = 0
    ls.symcounter = 0
    ls.isbroadcast = 0
    ls.register_size = 0
    ls.register_count = 0
    ls.cache_linesize = 0
    ls.cache_size = (0,0,0)
    ls.ureduct = -1
    ls.equalarraydims = Tuple{Vector{Symbol},Vector{Int}}[]
    ls.omop = OffsetLoadCollection()
    ls.loopordermap =  Int[]
    ls.loopindexesbit = Bool[]
    ls.mod = mod
    ls
end

"""
Used internally to create symbols unique for this loopset.
This is used so that identical loops will create identical `_avx_!` calls in the macroexpansions, hopefully reducing recompilation.
"""
gensym!(ls::LoopSet, s) = Symbol("###$(s)###$(ls.symcounter += 1)###")

function cacheunrolled!(ls::LoopSet, u₁loop::Symbol, u₂loop::Symbol, vloopsym::Symbol)
    vloop = getloop(ls, vloopsym)
    for op ∈ operations(ls)
        empty!(children(op))
        for opp ∈ parents(op)
            push!(children(opp), op)
        end
    end
    for op ∈ operations(ls)
        setunrolled!(ls, op, u₁loop, u₂loop, vloopsym)
        if accesses_memory(op)
            rc = rejectcurly(ls, op, u₁loop, vloopsym)
            op.rejectcurly = rc
            if rc
                op.rejectinterleave = true
            else
                omop = ls.omop
                batchid, opind = omop.batchedcollectionmap[identifier(op)]
                op.rejectinterleave = ((batchid == 0) || (!isvectorized(op))) || rejectinterleave(ls, op, vloop, omop.batchedcollections[batchid])
            end
        end
    end
end
function setunrolled!(ls::LoopSet, op::Operation, u₁loopsym::Symbol, u₂loopsym::Symbol, vectorized::Symbol)
  u₁::Bool = u₂::Bool = v::Bool = false
  for ld ∈ loopdependencies(op)
    u₁ |= ld === u₁loopsym
    u₂ |= ld === u₂loopsym
    v  |= ld === vectorized
  end
  if isconstant(op)
    for opp ∈ children(op)
      u₁ = u₁ &&  u₁loopsym ∈ loopdependencies(opp)
      u₂ = u₂ &&  u₂loopsym ∈ loopdependencies(opp)
      v  = v  && vectorized ∈ loopdependencies(opp)
    end
    if isouterreduction(ls, op) ≠ -1 && !all((u₁,u₂,v))
      opv = true
      for opp ∈ parents(op)
        if iscompute(opp) && instruction(opp).instr ≢ :identity
          opv = false
          break
        end
      end
      if opv
        if !u₁ && u₁loopsym ∈ reduceddependencies(op)
          u₁ = true
        end
        if !u₂ && u₂loopsym ∈ reduceddependencies(op)
          u₂ = true
        end
        if !v && vectorized ∈ reduceddependencies(op)
          v = true
        end
      end
    end
  end
  op.u₁unrolled = u₁
  op.u₂unrolled = u₂
  op.vectorized = v
  nothing
end

rejectcurly(op::Operation) = op.rejectcurly
rejectinterleave(op::Operation) = op.rejectinterleave
num_loops(ls::LoopSet) = length(ls.loops)
function oporder(ls::LoopSet)
    N = length(ls.loop_order.loopnames)
    reshape(ls.loop_order.oporder, (2,2,2,N))
end
names(ls::LoopSet) = ls.loop_order.loopnames
reversenames(ls::LoopSet) = ls.loop_order.bestorder
function getloopid_or_nothing(ls::LoopSet, s::Symbol)
    # @show ls.loopsymbols, s
    for (loopnum,sym) ∈ enumerate(ls.loopsymbols)
        s === sym && return loopnum
    end
end

getloopid(ls::LoopSet, s::Symbol) = getloopid_or_nothing(ls, s)::Int
# function getloopid(ls::LoopSet, s::Symbol)::Int
#     @show ls.loops
#     id = getloopid_or_nothing(ls, s)
#     @show id
#     id
# end
# getloop(ls::LoopSet, i::Integer) = getloop(ls, names(ls)[i])
getloop(ls::LoopSet, i::Integer) = ls.loops[ls.loopordermap[i]] # takes nest level after reordering
getloop_from_id(ls::LoopSet, i::Integer) = ls.loops[i] # takes w/ respect to original loop order.
getloop(ls::LoopSet, s::Symbol) = getloop_from_id(ls, getloopid(ls, s))
getloopsym(ls::LoopSet, i::Integer) = ls.loopsymbols[i]
Base.length(ls::LoopSet, s::Symbol) = length(getloop(ls, s))
function init_loop_map!(ls::LoopSet)
    @unpack loopordermap = ls
    order = names(ls)
    resize!(loopordermap, length(order))
    for (i,o) ∈ enumerate(order)
        loopordermap[i] = getloopid(ls,o)
    end
    nothing
end

# isstaticloop(ls::LoopSet, s::Symbol) = isstaticloop(getloop(ls,s))
# looprangehint(ls::LoopSet, s::Symbol) = length(getloop(ls, s))
# looprangesym(ls::LoopSet, s::Symbol) = getloop(ls, s).rangesym

"""
getop only works while construction a LoopSet object. You cannot use it while lowering.
"""
getop(ls::LoopSet, var::Number, elementbytes) = add_constant!(ls, var, elementbytes)
function getop(ls::LoopSet, var::Symbol, elementbytes::Int)
    get!(ls.opdict, var) do
        add_constant!(ls, var, elementbytes)
    end
end
function getop(ls::LoopSet, var::Symbol, deps, elementbytes::Int)
    get!(ls.opdict, var) do
        add_constant!(ls, var, deps, gensym!(ls, "constant"), elementbytes)
    end
end
getop(ls::LoopSet, i::Int) = ls.operations[i]

findop(ls::LoopSet, s::Symbol) = findop(operations(ls), s)
function findop(ops::Vector{Operation}, s::Symbol)
    for op ∈ ops
        name(op) === s && return op
    end
    throw(ArgumentError("Symbol $s not found."))
end

# """
# Returns an operation with the same name as `s`.
# """
# function getoperation(ls::LoopSet, s::Symbol)
#     for op ∈ Iterators.Reverse(operations(ls))
#         name(op) === s && return op
#     end
#     throw("Symbol $s not found among operations(ls).")
# end

function Operation(
    ls::LoopSet, variable, elementbytes, instruction,
    node_type, dependencies, reduced_deps, parents, ref = NOTAREFERENCE
)
    Operation(
        length(operations(ls)), variable, elementbytes, instruction,
        node_type, dependencies, reduced_deps, parents, ref
    )
end
function Operation(ls::LoopSet, variable, elementbytes, instr, optype, mpref::ArrayReferenceMetaPosition)
    Operation(length(operations(ls)), variable, elementbytes, instr, optype, mpref)
end

operations(ls::LoopSet) = ls.operations
function pushop!(ls::LoopSet, op::Operation, var::Symbol = name(op))
  if iscompute(op) && length(loopdependencies(op)) == 0
    op.node_type = constant
    opdef = callexpr(instruction(op))
    opparents = parents(op)
    mangledname = Symbol('#', instruction(op).instr, '#')
    while length(opparents) > 0
      oppname = name(popfirst!(opparents))
      mangledname = Symbol(mangledname, oppname, '#')
      push!(opdef.args, oppname)
      # if opp.instruction == LOOPCONSTANT 
      #   push!(opdef.args, name(opp))
      # else

      # end
    end
    op.mangledvariable = mangledname
    pushpreamble!(ls, Expr(:(=), name(op), opdef))
    op.instruction = LOOPCONSTANT
    push!(ls.preamble_symsym, (identifier(op), name(op)))
  end
  for opp ∈ operations(ls)
    if matches(op, opp)
      ls.opdict[var] = opp
      return opp
    end
  end
  push!(ls.operations, op)
  ls.opdict[var] = op
  op
end
function add_block!(ls::LoopSet, ex::Expr, elementbytes::Int, position::Int)
    for x ∈ ex.args
        x isa Expr || continue # be that general?
        x.head === :inbounds && continue
        push!(ls, x, elementbytes, position)
    end
end
function maybestatic!(expr::Expr)
    if expr.head === :call
        f = first(expr.args)
        if f === :length
            expr.args[1] = lv(:maybestaticlength)
        elseif f === :size && length(expr.args) == 3
            i = expr.args[3]
            if i isa Integer
                expr.args[1] = lv(:maybestaticsize)
                expr.args[3] = Expr(:call, Expr(:curly, :Val, convert(Int, i)))
            end
        else
            static_literals!(expr)
        end
    end
    expr
end
add_loop_bound!(ls::LoopSet, itersym::Symbol, bound::Union{Integer,Symbol}, upper::Bool, step::Bool)::MaybeKnown = MaybeKnown(bound, upper ? 1024 : 1)
function add_loop_bound!(ls::LoopSet, itersym::Symbol, bound::Expr, upper::Bool, step::Bool)::MaybeKnown
    maybestatic!(bound)
    N = gensym!(ls, string(itersym) * (upper ? "_loop_upper_bound" : (step ? "_loop_step" : "_loop_lower_bound")))
    pushprepreamble!(ls, Expr(:(=), N, bound))
    MaybeKnown(N, upper ? 1024 : 1)
end
static_literals!(s::Symbol) = s
function static_literals!(q::Expr)
    for (i,ex) ∈ enumerate(q.args)
        if ex isa Number
            q.args[i] = staticexpr(ex)
        elseif ex isa Expr
            static_literals!(ex)
        end
    end
    q
end
function range_loop!(ls::LoopSet, itersym::Symbol, l::MaybeKnown, u::MaybeKnown, s::MaybeKnown)
    rangename = gensym!(ls, "range"); lenname = gensym!(ls, "length")
    range = Expr(:call, :(:))
    pushexpr!(range, l)
    isone(s) || pushexpr!(range, s)
    pushexpr!(range, u)
    pushprepreamble!(ls, Expr(:(=), rangename, range))
    pushprepreamble!(ls, Expr(:(=), lenname, Expr(:call, lv(:maybestaticlength), rangename)))
    Loop(itersym, l, u, s, rangename, lenname)
end
function range_loop!(ls::LoopSet, r::Expr, itersym::Symbol)::Loop
    lower = r.args[2]
    sii::Bool = if length(r.args) == 3
        step = 1
        upper = r.args[3]
        true
    elseif length(r.args) == 4
        step = r.args[3]
        upper = r.args[4]
        isa(step, Integer)
    else
        throw("Literal ranges must have either 2 or 3 arguments.")
    end
    lii::Bool = lower isa Integer
    uii::Bool = upper isa Integer
    l::MaybeKnown = add_loop_bound!(ls, itersym, lower, false, false)
    u::MaybeKnown = add_loop_bound!(ls, itersym, upper, true, false)
    s::MaybeKnown = add_loop_bound!(ls, itersym, step, false, true)
    range_loop!(ls, itersym, l, u, s)
end
function oneto_loop!(ls::LoopSet, r::Expr, itersym::Symbol)::Loop
    otN = r.args[2]
    l = MaybeKnown(1, 0)
    s = MaybeKnown(1, 0)
    u::MaybeKnown = if otN isa Integer
        rangename = lensym = Symbol("")
        MaybeKnown(convert(Int, otN)::Int, 0)
    else
        otN isa Expr && maybestatic!(otN)
        lensym = N = gensym!(ls, "loop" * string(itersym))
        rangename = gensym!(ls, "range");
        pushprepreamble!(ls, Expr(:(=), N, otN))
        pushprepreamble!(ls, Expr(:(=), rangename, Expr(:call, :(:), staticexpr(1), N)))
        MaybeKnown(N, 1024)
    end
    Loop(itersym, l, u, s, rangename, lensym)
end

@inline _reverse(r) = maybestaticlast(r):-static_step(r):maybestaticfirst(r)
@inline canonicalize_range(r::OptionallyStaticUnitRange) = r
@inline function canonicalize_range(r::OptionallyStaticRange, ::StaticInt{S}) where {S}
    ifelse(ArrayInterface.gt(StaticInt{S}(), Zero()), r, _reverse(r))
end
@inline canonicalize_range(r::OptionallyStaticRange, s::Integer) = s > 0 ? r : _reverse(r)
@inline canonicalize_range(r::CloseOpen) = r
@inline canonicalize_range(r::AbstractUnitRange) = maybestaticfirst(r):maybestaticlast(r)
@inline canonicalize_range(r::OptionallyStaticRange) = canonicalize_range(r, static_step(r))
@inline canonicalize_range(r::AbstractRange) = canonicalize_range(maybestaticfirst(r):static_step(r):maybestaticlast(r))
@inline canonicalize_range(r::CartesianIndices) = CartesianIndices(map(canonicalize_range, r.indices))

function misc_loop!(ls::LoopSet, r::Union{Expr,Symbol}, itersym::Symbol, staticstepone::Bool)::Loop
    rangename = gensym!(ls, "looprange" * string(itersym)); lenname = gensym!(ls, "looplen" * string(itersym));
    pushprepreamble!(ls, Expr(:(=), rangename, Expr(:call, lv(:canonicalize_range), :(@inbounds $(static_literals!(r))))))
    pushprepreamble!(ls, Expr(:(=), lenname, Expr(:call, lv(:maybestaticlength), rangename)))
    L = add_loop_bound!(ls, itersym, Expr(:call, lv(:maybestaticfirst), rangename), false, false)
    U = add_loop_bound!(ls, itersym, Expr(:call, lv(:maybestaticlast), rangename), true, false)
    if staticstepone
        Loop(itersym, L, U, MaybeKnown(1), rangename, lenname)
    else
        S = add_loop_bound!(ls, itersym, Expr(:call, lv(:static_step), rangename), false, true)
        Loop(itersym, L, U, S, rangename, lenname)
    end
end

function indices_loop!(ls::LoopSet, r::Expr, itersym::Symbol)::Loop
    if length(r.args) == 3
        arrays = r.args[2]
        dims = r.args[3]
        if isexpr(arrays, :tuple) && length(arrays.args) > 1 && all(s -> s isa Symbol, arrays.args)
            narrays =  length(arrays.args)::Int
            if dims isa Integer
                # ids = Vector{NTuple{2,Int}}(undef, narrays)
                vptrs = Vector{Symbol}(undef, narrays)
                mdims = fill(dims::Int, narrays)
                # _d::Int = dims
                for n ∈ 1:narrays
                    a_s::Symbol = arrays.args[n]
                    vptrs[n] = vptr(a_s)
                end
                push!(ls.equalarraydims, (vptrs, mdims))
            elseif isexpr(dims, :tuple) && length(dims.args) == narrays && all(i -> i isa Integer, dims.args)
                # ids = Vector{NTuple{2,Int}}(undef, narrays)
                vptrs = Vector{Symbol}(undef, narrays)
                mdims = Vector{Int}(undef, narrays)
                for n ∈ 1:narrays
                    a_s::Symbol = arrays.args[n]
                    vptrs[n] = vptr(a_s)
                    mdims[n] = dims.args[n]
                end
                push!(ls.equalarraydims, (vptrs, mdims))
                # push!(ls.equalarraydims, ids)
            end
        end
    end
    misc_loop!(ls, r, itersym, true)
end

"""
This function creates a loop, while switching from 1 to 0 based indices
"""
function register_single_loop!(ls::LoopSet, looprange::Expr)
    itersym = (looprange.args[1])::Symbol
    r = looprange.args[2]
    loop = if isexpr(r, :call)
        r = r::Expr        # julia#37342
        f = first(r.args)
        if f === :(:)
            range_loop!(ls, r, itersym)
        elseif f === :OneTo || isscopedname(f, :Base, :OneTo)
            oneto_loop!(ls, r, itersym)
        elseif f === :indices || (isexpr(f, :(.), 2) && (f.args[2] === QuoteNode(:indices)) && ((f.args[1] === :ArrayInterface) || (f.args[1] === :LoopVectorization)))
            indices_loop!(ls, r, itersym)
        else
            misc_loop!(ls, r, itersym, (f === :eachindex) | (f === :axes))
        end
    elseif isa(r, Symbol)
        misc_loop!(ls, r, itersym, false)
    else
        throw(LoopError("Unrecognized loop range type: $r."))
    end
    add_loop!(ls, loop, itersym)
    nothing
end
function register_loop!(ls::LoopSet, looprange::Expr)
    if looprange.head === :block # multiple loops
        for lr ∈ looprange.args
            register_single_loop!(ls, lr::Expr)
        end
    else
        @assert looprange.head === :(=)
        register_single_loop!(ls, looprange)
    end
end
function add_loop!(ls::LoopSet, q::Expr, elementbytes::Int)
    register_loop!(ls, q.args[1]::Expr)
    body = q.args[2]::Expr
    position = length(ls.loopsymbols)
    if body.head === :block
        add_block!(ls, body, elementbytes, position)
    else
        push!(ls, q, elementbytes, position)
    end
end
function add_loop!(ls::LoopSet, loop::Loop, itersym::Symbol = loop.itersymbol)
    push!(ls.loopsymbols, itersym)
    push!(ls.loops, loop)
    nothing
end

# function instruction(x)
#     x isa Symbol ? x : last(x.args).value
# end
# instruction(ls::LoopSet, f::Symbol) = instruction!(ls, f)
function instruction!(ls::LoopSet, x::Expr)
    # x isa Symbol && return x
    if x.head === :$
        _x = only(x.args)
        _x isa Symbol && return instruction!(ls, _x)
        @assert _x isa Expr
        x = _x
    end
    instr = last(x.args).value
    if instr ∉ keys(COST)
        instr = gensym!(ls, "f")
        pushpreamble!(ls, Expr(:(=), instr, x))
        Instruction(Symbol(""), instr)
    else
        Instruction(:LoopVectorization, instr)
    end
end
instruction!(ls::LoopSet, x::Symbol) = instruction(x)
function instruction!(ls::LoopSet, f::F) where {F <: Function}
    get(FUNCTIONSYMBOLS, F) do
        instr = gensym!(ls, "f")
        pushpreamble!(ls, Expr(:(=), instr, f))
        Instruction(Symbol(""), instr)
    end
end


function maybe_const_compute!(ls::LoopSet, LHS::Symbol, op::Operation, elementbytes::Int, position::Int)
    # return op
    if iscompute(op) && iszero(length(loopdependencies(op)))
        ls.opdict[LHS] = add_constant!(ls, LHS, ls.loopsymbols[1:position], gensym!(ls, instruction(op).instr), elementbytes, :numericconstant)
    else
        op
    end
end
strip_op_linenumber_nodes(q::Expr) = only(filter(x -> !isa(x, LineNumberNode), q.args))

function add_operation!(ls::LoopSet, LHS::Symbol, RHS::Symbol, elementbytes::Int, position::Int)
    add_constant!(ls, RHS, ls.loopsymbols[1:position], LHS, elementbytes)
end
function add_operation!(
    ls::LoopSet, LHS::Symbol, RHS::Expr, elementbytes::Int, position::Int
)
    if RHS.head === :ref
        add_load_ref!(ls, LHS, RHS, elementbytes)
    elseif RHS.head === :call
        f = first(RHS.args)
        if f === :getindex
            add_load_getindex!(ls, LHS, RHS, elementbytes)
        elseif f === :zero || f === :one
            c = gensym!(ls, f)
            op = add_constant!(ls, c, ls.loopsymbols[1:position], LHS, elementbytes, :numericconstant)
            if f === :zero
                push!(ls.preamble_zeros, (identifier(op), IntOrFloat))
            else
                push!(ls.preamble_funcofeltypes, (identifier(op), MULTIPLICATIVE_IN_REDUCTIONS))
            end
            op
        else
            # maybe_const_compute!(ls, add_compute!(ls, LHS, RHS, elementbytes, position), elementbytes, position)
            add_compute!(ls, LHS, RHS, elementbytes, position)
        end
    elseif RHS.head === :if
        add_if!(ls, LHS, RHS, elementbytes, position)
    elseif RHS.head === :block
        add_operation!(ls, LHS, strip_op_linenumber_nodes(RHS), elementbytes, position)
    elseif RHS.head === :(.)
        c = gensym!(ls, "getproperty")
        pushpreamble!(ls, Expr(:(=), c, RHS))
        add_constant!(ls, c, elementbytes)
        # op = add_constant!(ls, c, ls.loopsymbols[1:position], LHS, elementbytes, :numericconstant)
        # pushpreamble!(ls, op, c)
        # op
    else
        throw(LoopError("Expression not recognized.", RHS))
    end
end
add_operation!(ls::LoopSet, RHS::Expr, elementbytes::Int, position::Int) = add_operation!(ls, gensym!(ls, "LHS"), RHS, elementbytes, position)
function add_operation!(
    ls::LoopSet, LHS_sym::Symbol, RHS::Expr, LHS_ref::ArrayReferenceMetaPosition, elementbytes::Int, position::Int
)
    if RHS.head === :ref# || (RHS.head === :call && first(RHS.args) === :getindex)
        array, rawindices = ref_from_expr!(ls, RHS)
        RHS_ref = array_reference_meta!(ls, array, rawindices, elementbytes, gensym!(ls, LHS_sym))
        op = add_load!(ls, RHS_ref, elementbytes)
        iop = add_compute!(ls, LHS_sym, :identity, [op], elementbytes)
        # pushfirst!(LHS_ref.parents, iop)
    elseif RHS.head === :call
        f = first(RHS.args)
        if f === :getindex
            add_load!(ls, LHS_sym, LHS_ref, elementbytes)
        elseif f === :zero || f === :one
            c = gensym!(ls, f)
            op = add_constant!(ls, c, ls.loopsymbols[1:position], LHS_sym, elementbytes, :numericconstant)
            # op = add_constant!(ls, c, Symbol[], LHS_sym, elementbytes, :numericconstant)
            if f === :zero
                push!(ls.preamble_zeros, (identifier(op), IntOrFloat))
            else
                push!(ls.preamble_funcofeltypes, (identifier(op), MULTIPLICATIVE_IN_REDUCTIONS))
            end
            op
        else
            add_compute!(ls, LHS_sym, RHS, elementbytes, position, LHS_ref)
        end
    elseif RHS.head === :if
        add_if!(ls, LHS_sym, RHS, elementbytes, position, LHS_ref)
    elseif RHS.head === :block
        add_operation!(ls, LHS, strip_op_linenumber_nodes(RHS), elementbytes, position)
    elseif RHS.head === :(.)
        c = gensym!(ls, "getproperty")
        pushpreamble!(ls, Expr(:(=), c, RHS))
        add_constant!(ls, c, elementbytes)
        # op = add_constant!(ls, c, ls.loopsymbols[1:position], LHS_sym, elementbytes, :numericconstant)
        # pushpreamble!(ls, op, c)
        # op
    else
        throw(LoopError("Expression not recognized.", RHS))
    end
end

function prepare_rhs_for_storage!(ls::LoopSet, RHS::Union{Symbol,Expr}, array, rawindices, elementbytes::Int, position::Int)
    RHS isa Symbol && return add_store!(ls, RHS, array, rawindices, elementbytes)
    mpref = array_reference_meta!(ls, array, rawindices, elementbytes)
    cachedparents = copy(mpref.parents)
    ref = mpref.mref.ref
    lrhs = gensym!(ls, "RHS")
    mpref.varname = lrhs
    add_operation!(ls, lrhs, RHS, mpref, elementbytes, position)
    mpref.parents = cachedparents
    add_store!(ls, mpref, elementbytes)
end

function Base.push!(ls::LoopSet, ex::Expr, elementbytes::Int, position::Int)
    if ex.head === :call
        finex = first(ex.args)::Symbol
        if finex === :setindex!
            array, rawindices = ref_from_setindex!(ls, ex)
            prepare_rhs_for_storage!(ls, ex.args[3]::Union{Symbol,Expr}, array, rawindices, elementbytes, position)
        else
            error("Function $finex not recognized.")
        end
    elseif ex.head === :(=)
        LHS = ex.args[1]
        RHS = ex.args[2]
        if LHS isa Symbol
            if RHS isa Expr
                maybe_const_compute!(ls, LHS, add_operation!(ls, LHS, RHS, elementbytes, position), elementbytes, position)
            else
                add_constant!(ls, RHS, ls.loopsymbols[1:position], LHS, elementbytes)
            end
        elseif LHS isa Expr
            if LHS.head === :ref
                if RHS isa Symbol
                    add_store_ref!(ls, RHS, LHS, elementbytes)
                elseif RHS isa Expr
                    # need to check if LHS appears in RHS
                    # assign RHS to lrhs
                    array, rawindices = ref_from_expr!(ls, LHS)
                    prepare_rhs_for_storage!(ls, RHS, array, rawindices, elementbytes, position)
                else
                    add_store_ref!(ls, RHS, LHS, elementbytes)  # is this necessary? (Extension API?)
                end
            elseif LHS.head === :tuple
                @assert length(LHS.args) ≤ 9 "Functions returning more than 9 values aren't currently supported."
                lhstemp = gensym!(ls, "lhstuple")
                vparents = Operation[maybe_const_compute!(ls, lhstemp, add_operation!(ls, lhstemp, RHS, elementbytes, position), elementbytes, position)]
                for i ∈ eachindex(LHS.args)
                    f = (:first,:second,:third,:fourth,:fifth,:sixth,:seventh,:eighth,:ninth)[i]
                    lhsi = LHS.args[i]
                    if lhsi isa Symbol
                        add_compute!(ls, lhsi, f, vparents, elementbytes)
                    elseif lhsi isa Expr && lhsi.head === :ref
                        tempunpacksym = gensym!(ls, "tempunpack")
                        add_compute!(ls, tempunpacksym, f, vparents, elementbytes)
                        add_store_ref!(ls, tempunpacksym, lhsi, elementbytes)
                    else
                        throw(LoopError("Unpacking the above expression in the left hand side was not understood/supported.", lhsi))
                    end
                end
                first(vparents)
            else
                throw(LoopError("LHS not understood; only `:ref`s and `:tuple`s are currently supported.", LHS))
            end
        else
            throw(LoopError("LHS not understood.", LHS))
        end
    elseif ex.head === :block
        add_block!(ls, ex, elementbytes, position)
    elseif ex.head === :for
        add_loop!(ls, ex, elementbytes)
    elseif ex.head === :&&
        add_andblock!(ls, ex, elementbytes, position)
    elseif ex.head === :||
        add_orblock!(ls, ex, elementbytes, position)
    elseif ex.head === :local # Handle locals introduced by `@inbounds`; using `local` with `@avx` is not recomended (nor is `@inbounds`; which applies automatically regardless)
        @assert length(ex.args) == 1 # TODO replace assert + first with "only" once support for Julia < 1.4 is dropped
        localbody = first(ex.args)
        @assert localbody.head === :(=)
        @assert length(localbody.args) == 2
        LHS = (localbody.args[1])::Symbol
        RHS = push!(ls, (localbody.args[2]), elementbytes, position)
        if isstore(RHS)
            RHS
        else
            add_compute!(ls, LHS, :identity, [RHS], elementbytes)
        end
    else
        throw(LoopError("Don't know how to handle expression.", ex))
    end
end

function UnrollSpecification(ls::LoopSet, u₁loop::Symbol, u₂loop::Symbol, vloopsym::Symbol, u₁, u₂)
    order = names(ls)
    nu₁ = findfirst(Base.Fix2(===,u₁loop), order)::Int
    nu₂ = u₂ == -1 ? nu₁ : findfirst(Base.Fix2(===,u₂loop), order)::Int
    nv = findfirst(Base.Fix2(===,vloopsym), order)::Int
    UnrollSpecification(nu₁, nu₂, nv, u₁, u₂)
end

"""
  looplengthprod(ls::LoopSet)

Convert to `Float64` for the sake of non-64 bit platforms.
"""
function looplengthprod(ls::LoopSet)
    l = 1.0
    for loop ∈ ls.loops
        l *= Float64(length(loop))
    end
    l
end
    # prod(Float64 ∘ length, ls.loops)


function looplength(ls::LoopSet, s::Symbol)
    # search_tree(parents(operations(ls)[i]), name(op)) && return true
    id = getloopid_or_nothing(ls, s)
    if id === nothing
        l = 0.0
        # TODO: we could double count a loop.
        for op ∈ operations(ls)
            name(op) === s || continue
            for opp ∈ parents(op)
                if isloopvalue(opp)
                    oppname = first(loopdependencies(opp))
                    l += looplength(ls, oppname)
                elseif iscompute(opp)
                    oppname = name(opp)
                    l += looplength(ls, oppname)
                    # TODO elseif isconstant(opp)
                end
            end
            l += 1 - length(parents(op))
        end
        l
    else
        Float64(length(ls.loops[id]))
    end
end

offsetloadcollection(ls::LoopSet) = ls.omop
function fill_offset_memop_collection!(ls::LoopSet)
    omop = offsetloadcollection(ls)
    ops = operations(ls)
    num_ops = length(ops)
    @unpack opids, opidcollectionmap, batchedcollections, batchedcollectionmap = omop
    length(opidcollectionmap) == 0 || return
    resize!(opidcollectionmap, num_ops)
    fill!(opidcollectionmap, (0,0));
    resize!(batchedcollectionmap, num_ops)
    fill!(batchedcollectionmap, (0,0))
    empty!(opids);# empty!(offsets);
    for i ∈ 1:num_ops
        op = ops[i]
        isconditionalmemop(op) && continue # not supported yet
        opref = op.ref.ref
        opisload = isload(op)
        (opisload | isstore(op)) || continue
        opidcollectionmap[i] === (0,0) || continue # if not -1, we already handled
        isdiscontiguous(op) && continue
        collectionsize = 0
        for j ∈ i+1:num_ops
            opp = ops[j]
            isconditionalmemop(opp) && continue # not supported yet
            if opisload # Each collection is either entirely loads or entirely stores
                isload(opp) || continue
            else
                isstore(opp) || continue
            end
            # @show op opp
            oppref = opp.ref.ref
            sameref(opref, oppref) || continue
            if collectionsize == 0
                push!(opids, [identifier(op), identifier(opp)])
                # push!(offsets, [opref.offsets, oppref.offsets])
                opidcollectionmap[identifier(op)] = (length(opids),1)
            else
                push!(last(opids), identifier(opp))
                # push!(last(offsets), oppref.offsets)
            end
            opidcollectionmap[identifier(opp)] = (length(opids),length(last(opids)))
            collectionsize += 1
        end
    end
    for (collectionid,opidc) ∈ enumerate(opids)
        length(opidc) > 1 || continue

        # we check if we can turn the offsets into an unroll
        # we have up to `length(opidc)` loads to do, so we allocate that many "base" vectors
        # then we iterate through them, adding them to collections as appropriate
        # inner vector tuple is of (op_pos_w/in collection,o)
        unroll_collections = Vector{Vector{Tuple{Int,Int}}}(undef, length(opidc))
        num_unroll_collections = 0
        # num_ops_considered = length(opidc)
        r = 2:length(getindices(ops[first(opidc)]))

        for (i,opid) ∈ enumerate(opidc)
            op = ops[opid]
            offset = getoffsets(op)
            o = offset[1]
            v = view(offset, r)
            found_match = false
            for j ∈ 1:num_unroll_collections
                collectionⱼ = unroll_collections[j]
                # giet id (`first`) of first item in collection to get base offsets for comparison
                # @show op, opid ops[opidc[first(first(collectionⱼ))], first(first(collectionⱼ))
                if view(getoffsets(ops[opidc[first(first(collectionⱼ))]]), r) == v
                    found_match = true
                    push!(collectionⱼ, (i, o))
                end
            end
            # @show opid, found_match
            if !found_match
                num_unroll_collections += 1 # the `i` points to position within `opidc`
                unroll_collections[num_unroll_collections] = [(i,o)]
            end
        end
        for j ∈ 1:num_unroll_collections
            collectionⱼ = unroll_collections[j]
            collen = length(collectionⱼ)
            collen ≤ 1 && continue
            # we have multiple, easiest to process if we sort them
            sort!(collectionⱼ, by=last)
            istart = 1; ostart = last(first(collectionⱼ))
            oprev = ostart
            for i ∈ 2:collen
                onext = last(collectionⱼ[i])
                if onext == oprev + 1
                    oprev = onext
                    continue
                end
                # we skipped one, so we must now lower all previous
                if oprev ≠ ostart # it's just 1
                    pushbatchedcollection!(batchedcollections, batchedcollectionmap, opidc, ops, collectionⱼ, istart, i-1)
                end
                # restart istart and ostart
                istart = i
                ostart = onext
                oprev = onext
            end
            if istart ≠ collen
                pushbatchedcollection!(batchedcollections, batchedcollectionmap, opidc, ops, collectionⱼ, istart, collen)
            end
        end
    end
end

function pushbatchedcollection!(batchedcollections, batchedcollectionmap, opidc, ops, collectionⱼ, istart, istop)
    colview = view(collectionⱼ, istart:istop)
    push!(batchedcollections, colview)
    bclen = length(batchedcollections)
    for (i,(k,_)) ∈ enumerate(colview)
        # batchedcollectionmap[identifier(op)] gives index into `batchedcollections` containing `colview`
        batchedcollectionmap[identifier(ops[opidc[k]])] = (bclen,i)
    end
end

"""
Returns `0` if the op is the declaration of the constant outerreduction variable.
Returns `n`, where `n` is the constant declarations's index among parents(op), if op is an outter reduction.
Returns `-1` if not an outerreduction.
"""
function isouterreduction(ls::LoopSet, op::Operation)
  if isconstant(op) # equivalent to checking if length(loopdependencies(op)) == 0
    instr = op.instruction
    instr == LOOPCONSTANT && return 0
    instr.mod === GLOBALCONSTANT && return -1
    ops = operations(ls)
    for or ∈ ls.outer_reductions
      name(op) === name(ops[or]) && return 0
    end
    -1
  elseif iscompute(op)
    var = op.variable
    for opid ∈ ls.outer_reductions
      rop = operations(ls)[opid]
      if rop === op
        for (n,opp) ∈ enumerate(parents(op))
          opp.variable === var && return n
        end
      else
        for (n,opp) ∈ enumerate(parents(op))
          opp === rop && return n
          search_tree(parents(opp), rop.variable) && return n
        end
      end
    end
    -1
  else
    -1
  end
end

struct LoopError <: Exception
    msg
    ex
    LoopError(msg, ex=nothing) = new(msg, ex)
end

function Base.showerror(io::IO, err::LoopError)
    printstyled(io, err.msg; color = :red)
    err.ex === nothing || printstyled(io, '\n', err.ex)
end
