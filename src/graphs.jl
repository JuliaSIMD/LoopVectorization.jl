
struct UnrollSymbols
    u₁loopsym::Symbol
    u₂loopsym::Symbol
    vectorized::Symbol
end
struct UnrollArgs{T <: Union{Nothing,Int}}
    u₁::Int
    u₁loopsym::Symbol
    u₂loopsym::Symbol
    vectorized::Symbol
    u₂max::Int
    suffix::T
end
function UnrollArgs(u₁::Int, unrollsyms::UnrollSymbols, u₂max::Int, suffix)
    @unpack u₁loopsym, u₂loopsym, vectorized = unrollsyms
    UnrollArgs(u₁, u₁loopsym, u₂loopsym, vectorized, u₂max, suffix)
end
function UnrollArgs(ua::UnrollArgs, u::Int)
    @unpack u₁loopsym, u₂loopsym, vectorized, u₂max, suffix = ua
    UnrollArgs(u, u₁loopsym, u₂loopsym, vectorized, u₂max, suffix)
end
# UnrollSymbols(ua::UnrollArgs) = UnrollSymbols(ua.u₁loopsym, ua.u₂loopsym, ua.vectorized)

# isfirst(ua::UnrollArgs{Nothing}) = iszero(ua.u₁)
# isfirst(ua::UnrollArgs{Int}) = iszero(ua.u₁) & iszero(ua.suffix)

struct UnrollSpecification
    u₁loopnum::Int
    u₂loopnum::Int
    vectorizedloopnum::Int
    u₁::Int
    u₂::Int
end
# UnrollSpecification(ls::LoopSet, u₁loop::Loop, vectorized::Symbol, u₁, u₂) = UnrollSpecification(ls, u₁loop.itersymbol, vectorized, u₁, u₂)
function UnrollSpecification(us::UnrollSpecification, u₁, u₂)
    @unpack u₁loopnum, u₂loopnum, vectorizedloopnum = us
    UnrollSpecification(u₁loopnum, u₂loopnum, vectorizedloopnum, u₁, u₂)
end
# function UnrollSpecification(us::UnrollSpecification; u₁ = us.u₁, u₂ = us.u₂)
#     @unpack u₁loopnum, u₂loopnum, vectorizedloopnum = us
#     UnrollSpecification(u₁loopnum, u₂loopnum, vectorizedloopnum, u₁, u₂)
# end
isunrolled1(us::UnrollSpecification, n::Int) = us.u₁loopnum == n
isunrolled2(us::UnrollSpecification, n::Int) = !isunrolled1(us, n) && us.u₂loopnum == n
isvectorized(us::UnrollSpecification, n::Int) = us.vectorizedloopnum == n
function unrollfactor(us::UnrollSpecification, n::Int)
    @unpack u₁loopnum, u₂loopnum, u₁, u₂ = us
    (u₁loopnum == n) ? u₁ : ((u₂loopnum == n) ? u₂ : 1)
end

struct Loop
    itersymbol::Symbol
    starthint::Int
    stophint::Int
    startsym::Symbol
    stopsym::Symbol
    rangesym::Symbol
    lensym::Symbol
    startexact::Bool
    stopexact::Bool
end
startstopint(s::Int, start) = s
startstopint(s::Symbol, start) = start ? 1 : 1024
startstopsym(s::Int) = Symbol("##UNDEFINED##")
startstopsym(s::Symbol) = s
function Loop(itersymbol::Symbol, start::Union{Int,Symbol}, stop::Union{Int,Symbol}, rangename::Symbol, lensym::Symbol)
    Loop(
        itersymbol, startstopint(start,true), startstopint(stop,false),
        startstopsym(start), startstopsym(stop), rangename, lensym, start isa Int, stop isa Int
    )
end
Base.length(loop::Loop) = 1 + loop.stophint - loop.starthint
isstaticloop(loop::Loop) = loop.startexact & loop.stopexact


function startloop(loop::Loop, itersymbol)
    startexact = loop.startexact
    if startexact
        Expr(:(=), itersymbol, loop.starthint - 1)
    else
        Expr(:(=), itersymbol, Expr(:call, lv(:staticm1), Expr(:call, lv(:unwrap), loop.startsym)))
    end
end
addexpr(ex, incr) = Expr(:call, lv(:vadd_fast), ex, incr)
function addexpr(ex, incr::Number)
    if iszero(incr)
        incr
    elseif incr > 0
        Expr(:call, lv(:vadd_fast), ex, incr)
    else
        Expr(:call, lv(:vsub_fast), ex, -incr)
    end
end
addexpr(ex::Number, incr::Number) = ex + incr
subexpr(ex, incr) = Expr(:call, lv(:vsub_fast), ex, incr)
subexpr(ex::Number, incr::Number) = ex - incr
subexpr(ex, incr::Number) = addexpr(ex,  -incr)

staticmulincr(ptr, incr) = Expr(:call, lv(:staticmul), Expr(:call, :eltype, ptr), incr)
callpointerforcomparison(sym) = Expr(:call, lv(:pointerforcomparison), sym)
function vec_looprange(loopmax, UF::Int, mangledname::Symbol, ptrcomp::Bool)
    if ptrcomp
        vec_looprange(loopmax, UF, callpointerforcomparison(mangledname), staticmulincr(mangledname, VECTORWIDTHSYMBOL))
    else
        vec_looprange(loopmax, UF, mangledname, VECTORWIDTHSYMBOL)
    end
end
function vec_looprange(loopmax, UF::Int, mangledname, W)
    if isone(UF)
        compexpr = subexpr(loopmax, W)
    else
        compexpr = subexpr(loopmax, Expr(:call, lv(:vmul_fast), W, UF))
    end
    Expr(:call, :≤, mangledname, compexpr)
end
# function vec_looprange(loopmax, UF::Int, mangledname, W)
#     incr = if isone(UF)
#         Expr(:call, lv(:vsub_fast), W, staticexpr(1))
#     else
#         Expr(:call, lv(:vsub_fast), Expr(:call, lv(:vmul_fast), W, UF), staticexpr(1))
#     end
#     compexpr = subexpr(loopmax, incr)
#     Expr(:call, :<, mangledname, compexpr)
# end

function looprange(stopcon, incr::Int, mangledname)
    if iszero(incr)
        Expr(:call, :≤, mangledname, stopcon)
    elseif isone(incr)
        Expr(:call, :<, mangledname, stopcon)
    else
        Expr(:call, :≤, mangledname, subexpr(stopcon, incr))
    end
end
function looprange(loop::Loop, incr::Int, mangledname)
    loop.stopexact ? looprange(loop.stophint, incr, mangledname) : looprange(loop.stopsym, incr, mangledname)
end
function terminatecondition(
    loop::Loop, us::UnrollSpecification, n::Int, mangledname::Symbol, inclmask::Bool, UF::Int = unrollfactor(us, n)
)
    if !isvectorized(us, n)
        looprange(loop, UF, mangledname)
    elseif inclmask
        looprange(loop, 1, mangledname)
    elseif loop.stopexact
        vec_looprange(loop.stophint, UF, mangledname, false) # may not be u₂loop
    else
        vec_looprange(loop.stopsym, UF, mangledname, false) # may not be u₂loop
    end
end
function incrementloopcounter(us::UnrollSpecification, n::Int, mangledname::Symbol, UF::Int = unrollfactor(us, n))
    if isvectorized(us, n)
        if isone(UF)
            Expr(:(=), mangledname, Expr(:call, lv(:vadd_fast), VECTORWIDTHSYMBOL, mangledname))
        else
            Expr(:(=), mangledname, Expr(:call, lv(:vadd_fast), Expr(:call, lv(:vmul_fast), VECTORWIDTHSYMBOL, staticexpr(UF)), mangledname))
        end
    else
        Expr(:(=), mangledname, Expr(:call, lv(:vadd_fast), mangledname, UF))
    end
end
function incrementloopcounter!(q, us::UnrollSpecification, n::Int, UF::Int = unrollfactor(us, n))
    if isvectorized(us, n)
        if isone(UF)
            push!(q.args, Expr(:call, lv(:Static), VECTORWIDTHSYMBOL))
        else
            push!(q.args, Expr(:call, lv(:vmul_fast), VECTORWIDTHSYMBOL, Expr(:call, Expr(:curly, lv(:Static), UF))))
        end
    else
        push!(q.args, staticexpr(UF))
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
Base.@propagate_inbounds Base.getindex(lo::LoopOrder, i...) = lo.oporder[LinearIndices(size(lo))[i...]]

@enum NumberType::Int8 HardInt HardFloat IntOrFloat INVALID

struct LoopStartStopManager
    terminators::Vector{Int}
    incrementedptrs::Vector{Vector{ArrayReferenceMeta}}
    uniquearrayrefs::Vector{ArrayReferenceMeta}
end
# Must make it easy to iterate
# outer_reductions is a vector of indices (within operation vectors) of the reduction operation, eg the vmuladd op in a dot product
# O(N) search is faster at small sizes
struct LoopSet
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
    preamble_symint::Vector{Tuple{Int,Int}}
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
    unrollspecification::Base.RefValue{UnrollSpecification}
    loadelimination::Base.RefValue{Bool}
    lssm::Base.RefValue{LoopStartStopManager}
    vector_width::Base.RefValue{Int}
    symcounter::Base.RefValue{Int}
    isbroadcast::Base.RefValue{Bool}
    register_size::Base.RefValue{Int}
    register_count::Base.RefValue{Int}
    cache_linesize::Base.RefValue{Int}
    # opmask_register::Base.RefValue{Bool}
    mod::Symbol
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
function set_hw!(ls::LoopSet, rs::Int, rc::Int, cls::Int)
    ls.register_size[] = rs
    ls.register_count[] = rc
    ls.cache_linesize[] = cls
    # ls.opmask_register[] = omr
end
available_registers() = ifelse(has_opmask_registers(), register_count(), register_count() - One())
function set_hw!(ls::LoopSet)
    rs = Int(register_size())
    rc = Int(available_registers())
    cls = Int(cache_linesize())
    # omr = Bool(VectorizationBase.has_opmask_registers())
    set_hw!(ls, rs, rc, cls)
end
reg_size(ls::LoopSet) = ls.register_size[]
reg_count(ls::LoopSet) = ls.register_count[]
cache_lnsze(ls::LoopSet) = ls.cache_linesize[]
# opmask_reg(ls::LoopSet) = ls.opmask_register[]

pushprepreamble!(ls::LoopSet, ex) = push!(ls.prepreamble.args, ex)
function pushpreamble!(ls::LoopSet, op::Operation, v::Symbol)
    if v !== mangledvar(op)
        push!(ls.preamble_symsym, (identifier(op),v))
    end
    nothing
end
function pushpreamble!(ls::LoopSet, op::Operation, v::Number)
    typ = v isa Integer ? HardInt : HardFloat
    id = identifier(op)
    if iszero(v)
        push!(ls.preamble_zeros, (id, typ))
    elseif isone(v)
        push!(ls.preamble_funcofeltypes, (id, MULTIPLICATIVE_IN_REDUCTIONS))
    elseif v isa Integer
        push!(ls.preamble_symint, (id, convert(Int,v)))
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
    LoopSet(
        Symbol[], [0], Loop[],
        Dict{Symbol,Operation}(),
        Operation[], [0],
        Int[],
        LoopOrder(),
        Expr(:block),Expr(:block),
        Tuple{Int,Symbol}[],
        Tuple{Int,Int}[],
        Tuple{Int,Float64}[],
        Tuple{Int,NumberType}[],Tuple{Int,Symbol}[],
        Symbol[], Symbol[], Symbol[],
        ArrayReferenceMeta[],
        Matrix{Float64}(undef, 4, 2), # cost_vec
        Matrix{Float64}(undef, 4, 2), # reg_pres
        Bool[], Bool[], Ref{UnrollSpecification}(),
        Ref(false), Ref{LoopStartStopManager}(),
        Ref(0), Ref(0), Ref(false),
        Ref(0), Ref(0), Ref(0), #Ref(false),# hw params
        mod
    )
end

"""
Used internally to create symbols unique for this loopset.
This is used so that identical loops will create identical `_avx_!` calls in the macroexpansions, hopefully reducing recompilation.
"""
gensym!(ls::LoopSet, s) = Symbol("###$(s)###$(ls.symcounter[] += 1)###")

function cacheunrolled!(ls::LoopSet, u₁loop, u₂loop, vectorized)
    foreach(op -> setunrolled!(op, u₁loop, u₂loop, vectorized), operations(ls))
    foreach(empty! ∘ children, operations(ls))
    for op ∈ operations(ls)
        for opp ∈ parents(op)
            push!(children(opp), op)
        end
    end
end

num_loops(ls::LoopSet) = length(ls.loops)
function oporder(ls::LoopSet)
    N = length(ls.loop_order.loopnames)
    reshape(ls.loop_order.oporder, (2,2,2,N))
end
names(ls::LoopSet) = ls.loop_order.loopnames
reversenames(ls::LoopSet) = ls.loop_order.bestorder
function getloopid_or_nothing(ls::LoopSet, s::Symbol)
    for (loopnum,sym) ∈ enumerate(ls.loopsymbols)
        s === sym && return loopnum
    end
end
getloopid(ls::LoopSet, s::Symbol) = getloopid_or_nothing(ls, s)::Int
getloop(ls::LoopSet, s::Symbol) = ls.loops[getloopid(ls, s)]
# getloop(ls::LoopSet, i::Integer) = ls.loops[i]
getloopsym(ls::LoopSet, i::Integer) = ls.loopsymbols[i]
Base.length(ls::LoopSet, s::Symbol) = length(getloop(ls, s))

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
        end
    end
    expr
end
function add_loop_bound!(ls::LoopSet, itersym::Symbol, bound, upper::Bool = true)
    (bound isa Symbol && upper) && return bound
    bound isa Expr && maybestatic!(bound)
    N = gensym!(ls, string(itersym) * (upper ? "_loop_upper_bound" : "_loop_lower_bound"))
    pushprepreamble!(ls, Expr(:(=), N, bound))
    N
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

function range_loop!(ls::LoopSet, r::Expr, itersym::Symbol)::Loop
    lower = r.args[2]
    upper = r.args[3]
    lii::Bool = lower isa Integer
    liiv::Int = lii ? convert(Int, lower::Integer) : 1
    uii::Bool = upper isa Integer
    if lii & uii # both are integers
        loop = Loop(itersym, liiv, convert(Int, upper::Integer)::Int, Symbol(""), Symbol(""))
    elseif lii # only lower bound is an integer
        rangename = gensym!(ls, "range"); lenname = gensym!(ls, "length")
        loop = if upper isa Symbol
            pushprepreamble!(ls, Expr(:(=), rangename, Expr(:call, :(:), staticexpr(liiv), upper)))
            Loop(itersym, liiv, upper, rangename, lenname)
        elseif upper isa Expr
            supper = add_loop_bound!(ls, itersym, upper, true)
            pushprepreamble!(ls, Expr(:(=), rangename, Expr(:call, :(:), staticexpr(liiv), supper)))
            Loop(itersym, liiv, supper, rangename, lenname)
        else
            supper = add_loop_bound!(ls, itersym, upper, true)
            pushprepreamble!(ls, Expr(:(=), rangename, Expr(:call, :(:), staticexpr(liiv), supper)))
            Loop(itersym, liiv, supper, rangename, lenname)
        end
        pushprepreamble!(ls, Expr(:(=), lenname, Expr(:call, lv(:maybestaticlength), rangename)))
    elseif uii # only upper bound is an integer
        uiiv = convert(Int, upper::Integer)::Int
        rangename = gensym!(ls, "range"); lenname = gensym!(ls, "length")
        slower = add_loop_bound!(ls, itersym, lower, false)
        pushprepreamble!(ls, Expr(:(=), rangename, Expr(:call, :(:), slower, staticexpr(uiiv))))
        loop = Loop(itersym, slower, uiiv, rangename, lenname)
        pushprepreamble!(ls, Expr(:(=), lenname, Expr(:call, lv(:maybestaticlength), rangename)))
    else # neither are integers
        L = add_loop_bound!(ls, itersym, lower, false)
        U = add_loop_bound!(ls, itersym, upper, true)
        rangename = gensym!(ls, "range"); lenname = gensym!(ls, "length")
        pushprepreamble!(ls, Expr(:(=), rangename, Expr(:call, :(:), L, U)))
        pushprepreamble!(ls, Expr(:(=), lenname, Expr(:call, lv(:maybestaticlength), rangename)))
        loop = Loop(itersym, L, U, rangename, lenname)
    end
    loop
end
function oneto_loop!(ls::LoopSet, r::Expr, itersym::Symbol)::Loop
    otN = r.args[2]
    loop = if otN isa Integer
        Loop(itersym, 1, Int(otN)::Int, Symbol(""), Symbol(""))
    else
        otN isa Expr && maybestatic!(otN)
        N = gensym!(ls, "loop" * string(itersym))
        rangename = gensym!(ls, "range");
        pushprepreamble!(ls, Expr(:(=), N, otN))
        pushprepreamble!(ls, Expr(:(=), rangename, Expr(:call, :(:), staticexpr(1), N)))
        Loop(itersym, 1, N, rangename, N)
    end
    loop
end

@inline canonicalize_range(r::OptionallyStaticUnitRange) = r
@inline canonicalize_range(r::CloseOpen) = r
@inline canonicalize_range(r::AbstractUnitRange) = maybestaticfirst(r):maybestaticlast(r)
@inline canonicalize_range(r::CartesianIndices) = CartesianIndices(map(canonicalize_range, r.indices))

function misc_loop!(ls::LoopSet, r::Union{Expr,Symbol}, itersym::Symbol)::Loop
    rangename = gensym!(ls, "looprange" * string(itersym)); lenname = gensym!(ls, "looplen" * string(itersym));
    pushprepreamble!(ls, Expr(:(=), rangename, Expr(:call, lv(:canonicalize_range), static_literals!(r))))
    pushprepreamble!(ls, Expr(:(=), lenname, Expr(:call, lv(:maybestaticlength), rangename)))
    L = add_loop_bound!(ls, itersym, Expr(:call, lv(:maybestaticfirst), rangename), false)
    U = add_loop_bound!(ls, itersym, Expr(:call, lv(:maybestaticlast), rangename), true)
    Loop(itersym, L, U, rangename, lenname)
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
        else
            misc_loop!(ls, r, itersym)
        end
    elseif isa(r, Symbol)
        misc_loop!(ls, r, itersym)
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
function strip_op_linenumber_nodes(q::Expr)
    filtered = filter(x -> !isa(x, LineNumberNode), q.args)
    if VERSION ≥ v"1.4"
        only(filtered)
    else
        @assert isone(length(filtered))
        first(filtered)
    end
end

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
    else
        throw(LoopError("Expression not recognized.", RHS))
    end
end

function prepare_rhs_for_storage!(ls::LoopSet, RHS::Union{Symbol,Expr}, array, rawindices, elementbytes::Int, position::Int)
    RHS isa Symbol && return add_store!(ls, RHS, array, rawindices, elementbytes)
    mpref = array_reference_meta!(ls, array, rawindices, elementbytes)
    cachedparents = copy(mpref.parents)
    ref = mpref.mref.ref
    # id = findfirst(r -> r == ref, ls.refs_aliasing_syms)
    # lrhs = id === nothing ? gensym(:RHS) : ls.syms_aliasing_refs[id]
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

function UnrollSpecification(ls::LoopSet, u₁loop::Symbol, u₂loop::Symbol, vectorized::Symbol, u₁, u₂)
    order = names(ls)
    nu₁ = findfirst(isequal(u₁loop), order)::Int
    nu₂ = u₂ == -1 ? nu₁ : findfirst(isequal(u₂loop), order)::Int
    nv = findfirst(isequal(vectorized), order)::Int
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
    if isnothing(id)
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

# function getunrolled(ls::LoopSet)
#     order = names(ls)
#     us = ls.unrollspecification[]
#     @unpack u₁loopnum, u₂loopnum = us
#     order[u₁loopnum], order[u₂loopnum]
# end


struct LoopError <: Exception
    msg
    ex
    LoopError(msg, ex=nothing) = new(msg, ex)
end

function Base.showerror(io::IO, err::LoopError)
    printstyled(io, err.msg; color = :red)
    isnothing(err.ex) || printstyled(io, '\n', err.ex)
end
