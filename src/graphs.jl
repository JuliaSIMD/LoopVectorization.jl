
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
    suffix::T
end
function UnrollArgs(U::Int, unrollsyms::UnrollSymbols, suffix)
    @unpack u₁loopsym, u₂loopsym, vectorized = unrollsyms
    UnrollArgs(U, u₁loopsym, u₂loopsym, vectorized, suffix)
end
function UnrollArgs(ua::UnrollArgs, u::Int)
    @unpack u₁loopsym, u₂loopsym, vectorized, suffix = ua
    UnrollArgs(u, u₁loopsym, u₂loopsym, vectorized, suffix)
end
# UnrollSymbols(ua::UnrollArgs) = UnrollSymbols(ua.u₁loopsym, ua.u₂loopsym, ua.vectorized)

isfirst(ua::UnrollArgs{Nothing}) = iszero(ua.u₁)
isfirst(ua::UnrollArgs{Int}) = iszero(ua.u₁) & iszero(ua.suffix)

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
    startexact::Bool
    stopexact::Bool
end
startstopint(s::Int, start) = s
startstopint(s::Symbol, start) = start ? 1 : 1024
startstopsym(s::Int) = Symbol("##UNDEFINED##")
startstopsym(s::Symbol) = s
function Loop(itersymbol::Symbol, start::Union{Int,Symbol}, stop::Union{Int,Symbol})
    Loop(
        itersymbol, startstopint(start,true), startstopint(stop,false),
        startstopsym(start), startstopsym(stop), start isa Int, stop isa Int
    )
end
Base.length(loop::Loop) = 1 + loop.stophint - loop.starthint
isstaticloop(loop::Loop) = loop.startexact & loop.stopexact
function startloop(loop::Loop, isvectorized, itersymbol)
    startexact = loop.startexact
    if isvectorized
        if startexact
            Expr(:(=), itersymbol, Expr(:call, lv(:_MM), VECTORWIDTHSYMBOL, loop.starthint))
        else
            Expr(:(=), itersymbol, Expr(:call, lv(:_MM), VECTORWIDTHSYMBOL, loop.startsym))
        end
    elseif startexact
        Expr(:(=), itersymbol, loop.starthint)
    else
        Expr(:(=), itersymbol, Expr(:call, lv(:unwrap), loop.startsym))
    end
end
function vec_looprange(loop::Loop, UF::Int, mangledname::Symbol)
    isunrolled = UF > 1
    incr = if isunrolled
        Expr(:call, lv(:valmuladd), VECTORWIDTHSYMBOL, UF, -2)
    else
        Expr(:call, lv(:valsub), VECTORWIDTHSYMBOL, 2)
    end
    if loop.stopexact # split for type stability
        Expr(:call, lv(:scalar_less), mangledname, Expr(:call, :-, loop.stophint, incr))
    else
        Expr(:call, lv(:scalar_less), mangledname, Expr(:call, :-, loop.stopsym, incr))
    end
end
function looprange(loop::Loop, incr::Int, mangledname::Symbol)
    incr = 2 - incr
    if iszero(incr)
        Expr(:call, lv(:scalar_less), mangledname, loop.stopexact ? loop.stophint : loop.stopsym)
    else
        Expr(:call, lv(:scalar_less), mangledname, loop.stopexact ? loop.stophint + incr : Expr(:call, :+, loop.stopsym, incr))
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
function incrementloopcounter(us::UnrollSpecification, n::Int, mangledname::Symbol, UF::Int = unrollfactor(us, n))
    if isvectorized(us, n)
        if UF == 1
            Expr(:(=), mangledname, Expr(:call, lv(:valadd), VECTORWIDTHSYMBOL, mangledname))
        else
            Expr(:+=, mangledname, Expr(:call, lv(:valmul), VECTORWIDTHSYMBOL, UF))
        end
    else
        Expr(:+=, mangledname, UF)
    end
end

# load/compute/store × isunrolled × istiled × pre/post loop × Loop number
struct LoopOrder <: AbstractArray{Vector{Operation},5}
    oporder::Vector{Vector{Operation}}
    loopnames::Vector{Symbol}
    bestorder::Vector{Symbol}
end
function LoopOrder(N::Int)
    LoopOrder(
        [ Operation[] for _ ∈ 1:8N ],
        Vector{Symbol}(undef, N), Vector{Symbol}(undef, N)
    )
end
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
    preamble_funcofeltypes::Vector{Tuple{Int,Symbol}}
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
    ps
end
function save_tilecost!(ls::LoopSet)
    @inbounds for i ∈ 1:4
        ls.cost_vec[i,1] = ls.cost_vec[i,2]
        ls.reg_pres[i,1] = ls.reg_pres[i,2]
    end
end

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
        push!(ls.preamble_funcofeltypes, (id, :one))
    elseif v isa Integer
        push!(ls.preamble_symint, (id, convert(Int,v)))
    else
        push!(ls.preamble_symfloat, (id, convert(Float64,v)))
    end
end
pushpreamble!(ls::LoopSet, ex::Expr) = push!(ls.preamble.args, ex)
function pushpreamble!(ls::LoopSet, op::Operation, RHS::Expr)
    c = gensym(:licmconst)
    if RHS.head === :call && first(RHS.args) === :zero
        push!(ls.preamble_zeros, (identifier(op), IntOrFloat))
    elseif RHS.head === :call && first(RHS.args) === :one
        push!(ls.preamble_funcofeltypes, (identifier(op), :one))
    else
        pushpreamble!(ls, Expr(:(=), c, RHS))
        pushpreamble!(ls, op, c)
    end
    nothing
end
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
        Int[],Int[],
        Symbol[], Symbol[], Symbol[],
        ArrayReferenceMeta[],
        Matrix{Float64}(undef, 4, 2),
        Matrix{Float64}(undef, 4, 2),
        Bool[], Bool[], Ref{UnrollSpecification}(), Ref(false), mod
    )
end

num_loops(ls::LoopSet) = length(ls.loops)
function oporder(ls::LoopSet)
    N = length(ls.loop_order.loopnames)
    reshape(ls.loop_order.oporder, (2,2,2,N))
end
names(ls::LoopSet) = ls.loop_order.loopnames
function getloopid(ls::LoopSet, s::Symbol)::Int
    for (loopnum,sym) ∈ enumerate(ls.loopsymbols)
        s === sym && return loopnum
    end
end
getloop(ls::LoopSet, s::Symbol) = ls.loops[getloopid(ls, s)]
getloop(ls::LoopSet, i::Integer) = ls.loops[i]
getloopsym(ls::LoopSet, i::Integer) = ls.loopsymbols[i]
Base.length(ls::LoopSet, s::Symbol) = length(getloop(ls, s))

isstaticloop(ls::LoopSet, s::Symbol) = isstaticloop(getloop(ls,s))
looprangehint(ls::LoopSet, s::Symbol) = length(getloop(ls, s))
looprangesym(ls::LoopSet, s::Symbol) = getloop(ls, s).rangesym
getop(ls::LoopSet, var::Number, elementbytes) = add_constant!(ls, var, elementbytes)
function getop(ls::LoopSet, var::Symbol, elementbytes::Int)
    get!(ls.opdict, var) do
        add_constant!(ls, var, elementbytes)
    end
end
function getop(ls::LoopSet, var::Symbol, deps, elementbytes::Int)
    get!(ls.opdict, var) do
        add_constant!(ls, var, deps, gensym(:constant), elementbytes)
    end
end
getop(ls::LoopSet, i::Int) = ls.operations[i]

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

# load_operations(ls::LoopSet) = ls.loadops
# compute_operations(ls::LoopSet) = ls.computeops
# store_operations(ls::LoopSet) = ls.storeops
# function operations(ls::LoopSet)
    # Base.Iterators.flatten((
        # load_operations(ls),
        # compute_operations(ls),
        # store_operations(ls)
    # ))
# end
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
    N = gensym(string(itersym) * (upper ? "_loop_upper_bound" : "_loop_lower_bound"))
    pushpreamble!(ls, Expr(:(=), N, bound))
    N
end

"""
This function creates a loop, while switching from 1 to 0 based indices
"""
function register_single_loop!(ls::LoopSet, looprange::Expr)
    itersym = (looprange.args[1])::Symbol
    r = looprange.args[2]
    if isexpr(r, :call)
        f = first(r.args)
        loop::Loop = if f === :(:)
            lower = r.args[2]
            upper = r.args[3]
            lii::Bool = lower isa Integer
            liiv::Int = lii ? convert(Int, lower) : 1
            uii::Bool = upper isa Integer
            if lii & uii # both are integers
                Loop(itersym, liiv, convert(Int, upper))
            elseif lii # only lower bound is an integer
                if upper isa Symbol
                    Loop(itersym, liiv, upper)
                elseif upper isa Expr
                    Loop(itersym, liiv, add_loop_bound!(ls, itersym, upper, true))
                else
                    Loop(itersym, liiv, add_loop_bound!(ls, itersym, upper, true))
                end
            elseif uii # only upper bound is an integer
                uiiv = convert(Int, upper)
                Loop(itersym, add_loop_bound!(ls, itersym, lower, false), uiiv)
            else # neither are integers
                L = add_loop_bound!(ls, itersym, lower, false)
                U = add_loop_bound!(ls, itersym, upper, true)
                Loop(itersym, L, U)
            end
        elseif f === :OneTo || isscopedname(f, :Base, :OneTo)
            otN = r.args[2]
            if otN isa Integer
                Loop(itersym, 1, otN)
            else
                otN isa Expr && maybestatic!(otN)
                N = gensym("loop" * string(itersym))
                pushpreamble!(ls, Expr(:(=), N, otN))
                Loop(itersym, 1, N)
            end
        else
            N = gensym("loop" * string(itersym))
            pushpreamble!(ls, Expr(:(=), N, Expr(:call, lv(:maybestaticrange), r)))
            L = add_loop_bound!(ls, itersym, Expr(:call, lv(:maybestaticfirst), N), false)
            U = add_loop_bound!(ls, itersym, Expr(:call, lv(:maybestaticlast), N), true)
            Loop(itersym, L, U)
        end
    elseif isa(r, Symbol)
        # Treat similar to `eachindex`
        N = gensym("loop" * string(itersym))
        pushpreamble!(ls, Expr(:(=), N, Expr(:call, lv(:maybestaticrange), r)))
        L = add_loop_bound!(ls, itersym, Expr(:call, lv(:maybestaticfirst), N), false)
        U = add_loop_bound!(ls, itersym, Expr(:call, lv(:maybestaticlast), N), true)
        loop = Loop(itersym, L, U)
    else
        throw("Unrecognized loop range type: $r.")
    end
    add_loop!(ls, loop, itersym)
    nothing
end
function register_loop!(ls::LoopSet, looprange::Expr)
    if looprange.head === :block # multiple loops
        for lr ∈ looprange.args
            register_single_loop!(ls, lr)
        end
    else
        @assert looprange.head === :(=)
        register_single_loop!(ls, looprange)
    end
end
function add_loop!(ls::LoopSet, q::Expr, elementbytes::Int)
    register_loop!(ls, q.args[1])
    body = q.args[2]
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

function instruction(x)
    x isa Symbol ? x : last(x.args).value
end
# instruction(ls::LoopSet, f::Symbol) = instruction!(ls, f)
function instruction!(ls::LoopSet, x::Expr)
    x isa Symbol && return x
    instr = last(x.args).value
    if instr ∉ keys(COST)
        instr = gensym(:f)
        pushpreamble!(ls, Expr(:(=), instr, x))
    end
    Instruction(Symbol(""), instr)
end
instruction!(ls::LoopSet, x::Symbol) = instruction(x)

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
            c = gensym(f)
            op = add_constant!(ls, c, ls.loopsymbols[1:position], LHS, elementbytes, :numericconstant)
            if f === :zero
                push!(ls.preamble_zeros, (identifier(op), IntOrFloat))
            else
                push!(ls.preamble_funcofeltypes, (identifier(op), :one))
            end
            op
        else
            add_compute!(ls, LHS, RHS, elementbytes, position)
        end
    elseif RHS.head === :if
        add_if!(ls, LHS, RHS, elementbytes, position)
    else
        throw("Expression not recognized:\n$RHS")
    end
end
add_operation!(ls::LoopSet, RHS::Expr, elementbytes::Int, position::Int) = add_operation!(ls, gensym(:LHS), RHS, elementbytes, position)
function add_operation!(
    ls::LoopSet, LHS_sym::Symbol, RHS::Expr, LHS_ref::ArrayReferenceMetaPosition, elementbytes::Int, position::Int
)
    if RHS.head === :ref# || (RHS.head === :call && first(RHS.args) === :getindex)
        array, rawindices = ref_from_expr!(ls, RHS)
        RHS_ref = array_reference_meta!(ls, array, rawindices, elementbytes, gensym(LHS_sym))
        op = add_load!(ls, RHS_ref, elementbytes)
        iop = add_compute!(ls, LHS_sym, :identity, [op], elementbytes)
        # pushfirst!(LHS_ref.parents, iop)
    elseif RHS.head === :call
        f = first(RHS.args)
        if f === :getindex
            add_load!(ls, LHS_sym, LHS_ref, elementbytes)
        elseif f === :zero || f === :one
            c = gensym(f)
            op = add_constant!(ls, c, ls.loopsymbols[1:position], LHS_sym, elementbytes, :numericconstant)
            if f === :zero
                push!(ls.preamble_zeros, (identifier(op), IntOrFloat))
            else
                push!(ls.preamble_funcofeltypes, (identifier(op), :one))
            end
            op
        else
            add_compute!(ls, LHS_sym, RHS, elementbytes, position, LHS_ref)
        end
    elseif RHS.head === :if
        add_if!(ls, LHS_sym, RHS, elementbytes, position, LHS_ref)
    else
        throw("Expression not recognized:\n$x")
    end
end

function prepare_rhs_for_storage!(ls::LoopSet, RHS::Symbol, array, rawindices, elementbytes::Int, position::Int)
    add_store!(ls, RHS, array, rawindices, elementbytes)
end
function prepare_rhs_for_storage!(ls::LoopSet, RHS::Expr, array, rawindices, elementbytes::Int, position::Int)
    mpref = array_reference_meta!(ls, array, rawindices, elementbytes)
    cachedparents = copy(mpref.parents)
    ref = mpref.mref.ref
    # id = findfirst(r -> r == ref, ls.refs_aliasing_syms)
    # lrhs = id === nothing ? gensym(:RHS) : ls.syms_aliasing_refs[id]
    lrhs = gensym(:RHS)
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
            prepare_rhs_for_storage!(ls, ex.args[3], array, rawindices, elementbytes, position)
        else
            throw("Function $finex not recognized.")
        end
    elseif ex.head === :(=)
        LHS = ex.args[1]
        RHS = ex.args[2]
        if LHS isa Symbol
            if RHS isa Expr
                add_operation!(ls, LHS, RHS, elementbytes, position)
            else
                add_constant!(ls, RHS, ls.loopsymbols[1:position], LHS, elementbytes)
            end
        elseif LHS isa Expr
            @assert LHS.head === :ref
            if RHS isa Symbol
                add_store_ref!(ls, RHS, LHS, elementbytes)
            elseif RHS isa Expr
                # need to check if LHS appears in RHS
                # assign RHS to lrhs
                array, rawindices = ref_from_expr!(ls, LHS)
                prepare_rhs_for_storage!(ls, RHS, array, rawindices, elementbytes, position)
            else
                add_store_ref!(ls, RHS, LHS, elementbytes)
            end
        else
            throw("LHS not understood:\n$LHS")
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
        throw("Don't know how to handle expression:\n$ex")
    end
end

function UnrollSpecification(ls::LoopSet, u₁loop::Symbol, u₂loop::Symbol, vectorized::Symbol, u₁, u₂)
    order = names(ls)
    nu₁ = findfirst(isequal(u₁loop), order)::Int
    nu₂ = u₂ == -1 ? nu₁ : findfirst(isequal(u₂loop), order)::Int
    nv = findfirst(isequal(vectorized), order)::Int
    UnrollSpecification(nu₁, nu₂, nv, u₁, u₂)
end