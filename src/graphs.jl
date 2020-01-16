
isdense(::Type{<:DenseArray}) = true

# """
# ShortVector{T} simply wraps a Vector{T}, but uses a different hash function that is faster for short vectors to support using it as the keys of a Dict.
# This hash function scales O(N) with length of the vectors, so it is slow for long vectors.
# """
# struct ShortVector{T} <: DenseVector{T}
#     data::Vector{T}
# end
# Base.@propagate_inbounds Base.getindex(x::ShortVector, I...) = x.data[I...]
# Base.@propagate_inbounds Base.setindex!(x::ShortVector, v, I...) = x.data[I...] = v
# ShortVector{T}(::UndefInitializer, N::Integer) where {T} = ShortVector{T}(Vector{T}(undef, N))
# @inbounds Base.length(x::ShortVector) = length(x.data)
# @inbounds Base.size(x::ShortVector) = size(x.data)
# @inbounds Base.strides(x::ShortVector) = strides(x.data)
# @inbounds Base.push!(x::ShortVector, v) = push!(x.data, v)
# @inbounds Base.append!(x::ShortVector, v) = append!(x.data, v)
# function Base.hash(x::ShortVector, h::UInt)
#     @inbounds for n ∈ eachindex(x)
#         h = hash(x[n], h)
#     end
#     h
# end
# function Base.isequal(a::ShortVector{T}, b::ShortVector{T}) where {T}
#     length(a) == length(b) || return false
#     @inbounds for i ∈ 1:length(a)
#         a[i] === b[i] || return false
#     end
#     true
# end
# Base.convert(::Type{Vector}, sv::ShortVector) = sv.data
# Base.convert(::Type{Vector{T}}, sv::ShortVector{T}) where {T} = sv.data



struct Loop
    itersymbol::Symbol
    starthint::Int
    stophint::Int
    startsym::Symbol
    stopsym::Symbol
    startexact::Bool
    stopexact::Bool
end
function Loop(itersymbol::Symbol, start::Int, stop::Int)
    Loop(itersymbol, start, stop, Symbol("##UNDEFINED##"), Symbol("##UNDEFINED##"), true, true)
end
function Loop(itersymbol::Symbol, start::Int, stop::Symbol)
    Loop(itersymbol, start, 1024, Symbol("##UNDEFINED##"), stop, true, false)
end
function Loop(itersymbol::Symbol, start::Symbol, stop::Int)
    Loop(itersymbol, -1, stop, start, Symbol("##UNDEFINED##"), false, true)
end
function Loop(itersymbol::Symbol, start::Symbol, stop::Symbol)
    Loop(itersymbol, -1, 1024, start, stop, false, false)
end
Base.length(loop::Loop) = loop.stophint - loop.starthint
isstaticloop(loop::Loop) = loop.startexact & loop.stopexact
function startloop(loop::Loop, isvectorized, W, itersymbol = loop.itersymbol)
    startexact = loop.startexact
    if isvectorized
        if startexact
            Expr(:(=), itersymbol, Expr(:call, lv(:_MM), W, loop.starthint))
        else
            Expr(:(=), itersymbol, Expr(:call, lv(:_MM), W, loop.startsym))
        end
    elseif startexact
        Expr(:(=), itersymbol, loop.starthint)
    else
        Expr(:(=), itersymbol, loop.startsym)
    end
end
function vec_looprange(loop::Loop, isunrolled::Bool, W::Symbol, U::Int)
    incr = if isunrolled
        Expr(:call, lv(:valmuladd), W, U, -1)
    else
        Expr(:call, lv(:valsub), W, 1)
    end
    if loop.stopexact # split for type stability
        Expr(:call, :<, loop.itersymbol, Expr(:call, :-, loop.stophint, incr))
    else
        Expr(:call, :<, loop.itersymbol, Expr(:call, :-, loop.stopsym, incr))
    end
end                       
function looprange(loop::Loop, incr::Int, mangledname::Symbol)
    incr -= 1#one(Int32)
    if iszero(incr)
        Expr(:call, :<, mangledname, loop.stopexact ? loop.stophint : loop.stopsym)
    else
        Expr(:call, :<, mangledname, loop.stopexact ? loop.stophint - incr : Expr(:call, :-, loop.stopsym, incr))
    end
end
function terminatecondition(
    loop::Loop, W::Symbol, U::Int, T::Int, isvectorized::Bool, isunrolled::Bool, istiled::Bool,
    mangledname::Symbol = loop.itersymbol, mask::Nothing = nothing
)
    if isvectorized
        vec_looprange(loop, isunrolled, W, U) # may not be tiled
    else
        looprange(loop, isunrolled ? U : (istiled ? T : 1), mangledname)
    end
end
function terminatecondition(
    loop::Loop, W::Symbol, U::Int, T::Int, isvectorized::Bool, isunrolled::Bool, istiled::Bool,
    mangledname::Symbol, mask::Symbol
)
    looprange(loop, 1, mangledname)
end

# load/compute/store × isunroled × istiled × pre/post loop × Loop number
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

# Must make it easy to iterate
# outer_reductions is a vector of indixes (within operation vectors) of the reduction operation, eg the vmuladd op in a dot product
struct LoopSet
    loops::Dict{Symbol,Loop} # sym === loops[sym].itersymbol
    opdict::Dict{Symbol,Operation}
    operations::Vector{Operation} # Split them to make it easier to iterate over just a subset
    outer_reductions::Vector{Int} # IDs of reduction operations that need to be reduced at end.
    loop_order::LoopOrder
    # stridesets::Dict{ShortVector{Symbol},ShortVector{Symbol}}
    preamble::Expr # TODO: add preamble to lowering
    prepreamble::Expr # TODO: add preamble to lowering
    includedarrays::Vector{Tuple{Symbol,Int}}
    syms_aliasing_refs::Vector{Symbol} # O(N) search is faster at small sizes
    refs_aliasing_syms::Vector{ArrayReference}
    cost_vec::Matrix{Float64}
    reg_pres::Matrix{Int}
    included_vars::Vector{Bool}
    place_after_loop::Vector{Bool}
    W::Symbol
    T::Symbol
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


# function op_to_ref(ls::LoopSet, op::Operation)
    # s = op.variable
    # id = findfirst(ls.syms_aliasing_regs)
    # @assert id !== nothing
    # ls.refs_aliasing_syms[id]
# end
pushpreamble!(ls::LoopSet, ex) = push!(ls.preamble.args, ex)
pushprepreamble!(ls::LoopSet, ex) = push!(ls.prepreamble.args, ex)

function includesarray(ls::LoopSet, array::Symbol)
    for (a,i) ∈ ls.includedarrays
        a === array && return i
    end
    -1
end
function LoopSet()
    LoopSet(
        Dict{Symbol,Loop}(),
        Dict{Symbol,Operation}(),
        Operation[],
        Int[],
        LoopOrder(),
        Expr(:block),Expr(:block),
        Tuple{Symbol,Int}[],
        Symbol[],
        ArrayReference[],
        Matrix{Float64}(undef, 4, 2),
        Matrix{Int}(undef, 4, 2),
        Bool[], Bool[], gensym(:W), gensym(:T)
    )
end
num_loops(ls::LoopSet) = length(ls.loops)
function oporder(ls::LoopSet)
    N = length(ls.loop_order.loopnames)
    reshape(ls.loop_order.oporder, (2,2,2,N))
end
names(ls::LoopSet) = ls.loop_order.loopnames
Base.length(ls::LoopSet, s::Symbol) = length(ls.loops[s])
isstaticloop(ls::LoopSet, s::Symbol) = isstaticloop(ls.loops[s])
looprangehint(ls::LoopSet, s::Symbol) = length(ls.loops[s])
looprangesym(ls::LoopSet, s::Symbol) = ls.loops[s].rangesym
# itersyms(ls::LoopSet) = keys(ls.loops)
function getop(ls::LoopSet, var::Symbol, elementbytes::Int = 8)
    get!(ls.opdict, var) do
        # might add constant
        op = add_constant!(ls, var, elementbytes)
        pushpreamble!(ls, Expr(:(=), mangledvar(op), var))
        op
    end
end
function getop(ls::LoopSet, var::Symbol, deps, elementbytes::Int = 8)
    get!(ls.opdict, var) do
        # might add constant
        op = add_constant!(ls, var, deps, gensym(:constant), Symbol(""), elementbytes)
        pushpreamble!(ls, Expr(:(=), mangledvar(op), var))
        op
    end
end
getop(ls::LoopSet, i::Int) = ls.operations[i + 1]

@inline extract_val(::Val{N}) where {N} = N

function Operation(
    ls::LoopSet, variable, elementbytes, instruction,
    node_type, dependencies, reduced_deps, parents, ref = NOTAREFERENCE
)
    Operation(
        length(operations(ls)), variable, elementbytes, instruction,
        node_type, dependencies, reduced_deps, parents, ref
    )
end
function Operation(ls::LoopSet, var, elementbytes, instr, optype, mpref::ArrayReferenceMetaPosition)
    Operation(length(operations(ls)), var, elementbytes, instr, optype, mpref)
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
        matches(op, opp) && return opp
    end
    push!(ls.operations, op)
    ls.opdict[var] = op
    op
end
function add_block!(ls::LoopSet, ex::Expr, elementbytes::Int = 8)
    for x ∈ ex.args
        x isa Expr || continue # be that general?
        push!(ls, x, elementbytes)
    end
end
"""
This function creates a loop, while switching from 1 to 0 based indices
"""
function register_single_loop!(ls::LoopSet, looprange::Expr)
    itersym = (looprange.args[1])::Symbol
    r = (looprange.args[2])::Expr
    @assert r.head === :call
    f = first(r.args)
    loop::Loop = if f === :(:)
        lower = r.args[2]
        upper = r.args[3]
        lii::Bool = lower isa Integer
        liiv::Int = lii ? (convert(Int, lower)-1) : 0
        uii::Bool = upper isa Integer
        if lii & uii # both are integers
            Loop(itersym, liiv, convert(Int, upper))
        elseif lii # only lower bound is an integer
            if upper isa Symbol
                Loop(itersym, liiv, upper)
            else
                N = gensym(Symbol(itersym, :_loop_upper_bound))
                pushpreamble!(ls, Expr(:(=), N, upper))
                Loop(itersym, liiv, N)
            end
        elseif uii # only upper bound is an integer
            uiiv = convert(Int, upper)
            L = gensym(Symbol(itersym, :_loop_lower_bound))
            pushpreamble!(ls, Expr(:(=), L, Expr(:call, :-, lower, 1)))
            Loop(itersym, L, uiiv)
        else # neither are integers
            L = gensym(Symbol(itersym, :_loop_lower_bound))
            pushpreamble!(ls, Expr(:(=), L, Expr(:call, :-, lower, 1)))
            if upper isa Symbol
                Loop(itersym, L, upper)
            else
                N = gensym(Symbol(itersym, :_loop_upper_bound))
                pushpreamble!(ls, Expr(:(=), N, upper))
                Loop(itersym, L, N)
            end
        end
    elseif f === :eachindex
        N = gensym(Symbol(:loop, itersym))
        pushpreamble!(ls, Expr(:(=), N, Expr(:call, :length, r.args[2])))
        Loop(itersym, 0, N)
    else
        throw("Unrecognized loop range type: $r.")
    end
    ls.loops[itersym] = loop
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
function add_loop!(ls::LoopSet, q::Expr, elementbytes::Int = 8)
    register_loop!(ls, q.args[1])
    body = q.args[2]
    if body.head === :block
        add_block!(ls, body, elementbytes)
    else
        push!(ls, q, elementbytes)
    end
end
function add_loop!(ls::LoopSet, loop::Loop)
    ls.loops[loop.itersym] = loop
end

function add_vptr!(ls::LoopSet, op::Operation)
    ref = op.ref
    indexed = name(ref)
    id = identifier(op)
    if includesarray(ls, indexed) < 0
        push!(ls.includedarrays, (indexed, id))
        pushpreamble!(ls, Expr(:(=), vptr(op), Expr(:call, lv(:stridedpointer), indexed)))
    end
    nothing
end
# function intersection(depsplus, ls)
    # deps = Symbol[]
    # for dep ∈ depsplus
        # dep ∈ ls && push!(deps, dep)
    # end
    # deps
# end

function array_reference_meta!(ls::LoopSet, array::Symbol, rawindices, elementbytes::Int = 8)
    indices = Vector{Union{Symbol,Int}}(undef, length(rawindices))
    loopedindex = fill(false, length(indices))
    parents = Operation[]
    loopdependencies = Symbol[]
    reduceddeps = Symbol[]
    loopset = keys(ls.loops)
    for i ∈ eachindex(indices)
        ind = rawindices[i]
        if ind isa Integer
            indices[i] = ind - 1
        elseif ind isa Symbol
            indices[i] = ind
            if ind ∈ loopset
                loopedindex[i] = true
                push!(loopdependencies, ind)
            end
        elseif ind isa Expr
            parent = add_operation!(ls, gensym(:indexpr), ind, elementbytes)
            pushparent!(parents, loopdependencies, reduceddeps, parent)
            # var = get(ls.opdict, ind, nothing)
            indices[i] = name(parent)#mangledvar(parent)
        else
            throw("Unrecognized loop index: $ind.")
        end
    end
    length(parents) == 0 || pushfirst!(indices, Symbol("##DISCONTIGUOUSSUBARRAY##"))
    mref = ArrayReferenceMeta(ArrayReference( array, indices ), loopedindex)
    ArrayReferenceMetaPosition(mref, parents, loopdependencies, reduceddeps)
end
function tryrefconvert(ls::LoopSet, ex::Expr, elementbytes::Int = 8)::Tuple{Bool,ArrayReferenceMetaPosition}
    ya, yinds = if ex.head === :ref
        ref_from_ref(ex)
    elseif ex.head === :call
        f = first(ex.args)
        if f === :getindex
            ref_from_getindex(ex)
        elseif f === :setindex!
            ref_from_setindex(ex)
        else
            return false, NOTAREFERENCEMP
        end
    else
        return false, NOTAREFERENCEMP
    end
    true, array_reference_meta!(ls, ya, yinds, elementbytes)
end

function add_load!(
    ls::LoopSet, var::Symbol, array::Symbol, rawindices, elementbytes::Int = 8
)
    mpref = array_reference_meta!(ls, array, rawindices, elementbytes)
    add_load!(ls, var, mpref, elementbytes)
end
function add_load!(
    ls::LoopSet, var::Symbol, mpref::ArrayReferenceMetaPosition, elementbytes::Int = 8
)
    ref = mpref.mref.ref
    # try to CSE
    id = findfirst(r -> r == ref, ls.refs_aliasing_syms)
    if id === nothing
        push!(ls.syms_aliasing_refs, var)
        push!(ls.refs_aliasing_syms, ref)
    else
        opp = getop(ls, ls.syms_aliasing_refs[id], elementbytes)
        return isstore(opp) ? getop(ls, first(parents(opp))) : opp
    end
    # else, don't
    op = Operation( ls, var, elementbytes, :getindex, memload, mpref )
    add_vptr!(ls, op)
    pushop!(ls, op, var)
end

# for use with broadcasting
function add_simple_load!(
    ls::LoopSet, var::Symbol, ref::ArrayReference, elementbytes::Int = 8
)
    # if ref.loaded[] == true
        # op = getop(ls, var, elementbytes)
        # @assert var === op.variable
        # return op
    # end
    # loopset = keys(ls.loops)
    # loopdeps = Symbol[s for s ∈ loopdependencies(ref) if (s isa Symbol && s ∈ loopset)]
    loopdeps = Symbol[s for s ∈ ref.indices]
    mref = ArrayReferenceMeta(
        ref, fill(true, length(loopdeps))
    )
    op = Operation(
        length(operations(ls)), var, elementbytes,
        :getindex, memload, loopdeps,
        NODEPENDENCY, NOPARENTS, mref
    )
    add_vptr!(ls, op)
    pushop!(ls, op, var)
end
function add_load_ref!(ls::LoopSet, var::Symbol, ex::Expr, elementbytes::Int = 8)
    array, rawindices = ref_from_ref(ex)
    add_load!(ls, var, array, rawindices, elementbytes)
end
function add_load_getindex!(ls::LoopSet, var::Symbol, ex::Expr, elementbytes::Int = 8)
    array, rawindices = ref_from_getindex(ex)
    add_load!(ls, var, array, rawindices, elementbytes)
end
function instruction(x)
    x isa Symbol ? x : last(x.args).value
end

function addsetv!(s::AbstractVector{T}, v::T) where {T}
    for sᵢ ∈ s
        sᵢ === v && return nothing
    end
    push!(s, v)
    nothing
end
function mergesetv!(s1::AbstractVector{T}, s2::AbstractVector{T}) where {T}
    for s ∈ s2
        addsetv!(s1, s)
    end
    nothing
end
function mergesetdiffv!(
    s1::AbstractVector{T},
    s2::AbstractVector{T},
    s3::AbstractVector{T}
) where {T}
    for s ∈ s2
        s ∉ s3 && addsetv!(s1, s)
    end
    nothing
end
function setdiffv!(s3::AbstractVector{T}, s1::AbstractVector{T}, s2::AbstractVector{T}) where {T}
    for s ∈ s1
        (s ∈ s2) || (s ∉ s3 && push!(s3, s))
    end
end
# if it is a literal, that literal has to have been assigned to var in the preamble.

function add_constant!(ls::LoopSet, var::Symbol, elementbytes::Int = 8)
    pushop!(ls, Operation(length(operations(ls)), var, elementbytes, LOOPCONSTANT, constant, NODEPENDENCY, Symbol[], NOPARENTS), var)
end
function add_constant!(ls::LoopSet, var, elementbytes::Int = 8)
    sym = gensym(:temp)
    op = Operation(length(operations(ls)), sym, elementbytes, LOOPCONSTANT, constant, NODEPENDENCY, Symbol[], NOPARENTS)
    pushpreamble!(ls, Expr(:(=), mangledvar(op), var))
    pushop!(ls, op, sym)
end
# This version has loop dependencies. var gets assigned to sym when lowering.
function add_constant!(ls::LoopSet, var::Symbol, deps::Vector{Symbol}, sym::Symbol = gensym(:constant), f::Symbol = Symbol(""), elementbytes::Int = 8)
    # length(deps) == 0 && push!(ls.preamble.args, Expr(:(=), sym, var))
    pushop!(ls, Operation(length(operations(ls)), sym, elementbytes, Instruction(f,var), constant, deps, NODEPENDENCY, NOPARENTS), sym)
end
function add_constant!(
    ls::LoopSet, var, deps::Vector{Symbol}, sym::Symbol = gensym(:constant), f::Symbol = Symbol(""), elementbytes::Int = 8
)
    sym2 = gensym(:temp) # hack, passing meta info here
    pushpreamble!(ls, Expr(:(=), sym2, var))
    pushop!(ls, Operation(length(operations(ls)), sym, elementbytes, Instruction(f, sym2), constant, deps, NODEPENDENCY, NOPARENTS), sym)
end
function pushparent!(parents::Vector{Operation}, deps::Vector{Symbol}, reduceddeps::Vector{Symbol}, parent::Operation)
    push!(parents, parent)
    mergesetdiffv!(deps, loopdependencies(parent), reduceddependencies(parent))
    if !(isload(parent) || isconstant(parent))
        mergesetv!(reduceddeps, reduceddependencies(parent))
    end
    nothing
end
function pushparent!(mpref::ArrayReferenceMetaPosition, parent::Operation)
    pushparent!(mpref.parents, mpref.loopdependencies, mpref.reduceddeps, parent)
end
function add_parent!(
    parents::Vector{Operation}, deps::Vector{Symbol}, reduceddeps::Vector{Symbol}, ls::LoopSet, var, elementbytes::Int = 8
)
    parent = if var isa Symbol
        getop(ls, var, elementbytes)
    elseif var isa Expr #CSE candidate
        add_operation!(ls, gensym(:temporary), var, elementbytes)
    else # assumed constant
        add_constant!(ls, var, elementbytes)
    end
    pushparent!(parents, deps, reduceddeps, parent)
end
function add_reduction!(
    parents::Vector{Operation}, deps::Vector{Symbol}, reduceddeps::Vector{Symbol}, ls::LoopSet, var::Symbol, elementbytes::Int = 8
)
    get!(ls.opdict, var) do
        add_constant!(ls, var, elementbytes)
    end
    # pushparent!(parents, deps, reduceddeps, parent)
end
function add_reduction_update_parent!(
    parents::Vector{Operation}, deps::Vector{Symbol}, reduceddeps::Vector{Symbol}, ls::LoopSet,
    var::Symbol, instr::Symbol, elementbytes::Int = 8
)
    parent = getop(ls, var, elementbytes)
    isloopconstant = parent.instruction === LOOPCONSTANT
    Instr = Instruction(instr)
    # if parent is not an outer reduction...
    if !isloopconstant
        # and parent is not a reduction_zero
        reduct_zero = REDUCTION_ZERO[Instr]
        reductcombine = REDUCTION_SCALAR_COMBINE[Instr].name
        reductsym = gensym(:reduction)
        reductinit = add_constant!(ls, Expr(:call, reduct_zero, ls.T), loopdependencies(parent), reductsym, reduct_zero, elementbytes)
        if isconstant(parent) && reduct_zero === parent.instruction.mod #we can use parent op as initialization.
            reductcombine = REDUCTION_COMBINETO[reductcombine]
        # else # we cannot use parent op as initialization.
        end
    else
        reductinit = parent
        reductsym = var
        reductcombine = Symbol("")
    end
    # mergesetv!(reduceddeps, deps)
    setdiffv!(reduceddeps, deps, loopdependencies(reductinit))
    mergesetv!(reduceddependencies(reductinit), reduceddeps)
    pushparent!(parents, deps, reduceddeps, reductinit)#parent) # deps and reduced deps will not be disjoint
    op = Operation(length(operations(ls)), reductsym, elementbytes, instr, compute, deps, reduceddeps, parents)
    parent.instruction === LOOPCONSTANT && push!(ls.outer_reductions, identifier(op))
    opout = pushop!(ls, op, var) # note this overwrites the entry in the operations dict, but not the vector
    isloopconstant && return opout
    # create child
    childdeps = Symbol[]; childrdeps = Symbol[]; childparents = Operation[]
    pushparent!(childparents, childdeps, childrdeps, op) # reduce op
    pushparent!(childparents, childdeps, childrdeps, parent) # to
    child = Operation(
        length(operations(ls)), name(parent), elementbytes, reductcombine, compute, childdeps, childrdeps, childparents
    )
    pushop!(ls, child, name(parent))
end
function add_compute!(
    ls::LoopSet, var::Symbol, ex::Expr, elementbytes::Int = 8,
    mpref::Union{Nothing,ArrayReferenceMetaPosition} = nothing
)
    @assert ex.head === :call
    instr = instruction(first(ex.args))::Symbol
    args = @view(ex.args[2:end])
    parents = Operation[]
    deps = Symbol[]
    reduceddeps = Symbol[]
    reduction = false
    for arg ∈ args
        if var === arg
            reduction = true
            add_reduction!(parents, deps, reduceddeps, ls, arg, elementbytes)
        elseif arg isa Expr
            isref, argref = tryrefconvert(ls, arg, elementbytes)
            if isref
                if mpref == argref
                    reduction = true
                    add_load!(ls, var, mpref, elementbytes)
                else
                    pushparent!(parents, deps, reduceddeps, add_load!(ls, gensym(:tempload), argref, elementbytes))
                end
            else
                add_parent!(parents, deps, reduceddeps, ls, arg, elementbytes)
            end
        else
            add_parent!(parents, deps, reduceddeps, ls, arg, elementbytes)
        end
    end
    if reduction # arg[reduction] is the reduction
        add_reduction_update_parent!(parents, deps, reduceddeps, ls, var, instr, elementbytes)
    else
        op = Operation(length(operations(ls)), var, elementbytes, instr, compute, deps, reduceddeps, parents)
        pushop!(ls, op, var)
    end
end
function add_unique_store!(ls::LoopSet, op::Operation)
    add_vptr!(ls, op)
    pushop!(ls, op, name(op.ref))
end
function cse_store!(ls::LoopSet, op::Operation)
    id = identifier(op)
    ls.operations[id] = op
    ls.opdict[op.variable] = op
    op
end
function add_store!(ls::LoopSet, op::Operation)
    nops = length(ls.operations)
    id = op.identifier
    id == nops ? add_unique_store!(ls, op) : cse_store!(ls, op)
end
function add_store!(
    ls::LoopSet, var::Symbol, mpref::ArrayReferenceMetaPosition, elementbytes::Int = 8
)
    parents = mpref.parents
    ldref = mpref.loopdependencies
    reduceddeps = mpref.reduceddeps
    parent = getop(ls, var, ldref, elementbytes)
    # pushfirst!(parents, parent)
    pvar = parent.variable
    nops = length(ls.operations)
    id = nops
    if pvar ∉ ls.syms_aliasing_refs
        push!(ls.syms_aliasing_refs, pvar)
        push!(ls.refs_aliasing_syms, mpref.mref.ref)
        # add_unique_store!(ls, mref, parents, ldref, reduceddeps, elementbytes)
    else
        # try to cse store
        # different from cse load, because the other op here must be a store
        ref = mpref.mref.ref
        for opp ∈ operations(ls)
            isstore(opp) || continue
            if ref == opp.ref.ref# && return cse_store!(ls, identifier(opp), mref, parents, ldref, reduceddeps, elementbytes)
                id = opp.identifier
            end
        end
        # add_unique_store!(ls, mref, parents, ldref, reduceddeps, elementbytes)        
    end
    pushparent!(parents, ldref, reduceddeps, parent)
    op = Operation( id, name(mpref), elementbytes, :setindex!, memstore, mpref )#loopdependencies, reduceddeps, parents, mpref.mref )
    add_store!(ls, op)
end
function add_store!(
    ls::LoopSet, var::Symbol, array::Symbol, rawindices, elementbytes::Int = 8
)
    mpref = array_reference_meta!(ls, array, rawindices, elementbytes)
    add_store!(ls, var, mpref, elementbytes)
end
function add_simple_store!(ls::LoopSet, var::Symbol, ref::ArrayReference, elementbytes::Int = 8)
    mref = ArrayReferenceMeta(
        ref, fill(true, length(getindices(ref)))
    )
    parents = [getop(ls, var, elementbytes)]
    ldref = convert(Vector{Symbol}, getindices(ref))
    op = Operation( ls, name(mref), elementbytes, :setindex!, memstore, ldref, NODEPENDENCY, parents, mref )
    add_unique_store!(ls, op)
end
function add_store_ref!(ls::LoopSet, var::Symbol, ex::Expr, elementbytes::Int = 8)
    array, raw_indices = ref_from_ref(ex)
    add_store!(ls, var, array, raw_indices, elementbytes)
end
function add_store_setindex!(ls::LoopSet, ex::Expr, elementbytes::Int = 8)
    array, raw_indices = ref_from_setindex(ex)
    add_store!(ls, (ex.args[2])::Symbol, array, rawindices, elementbytes)
end
# add operation assigns X to var
function add_operation!(
    ls::LoopSet, LHS::Symbol, RHS::Expr, elementbytes::Int = 8
)
    if RHS.head === :ref
        add_load_ref!(ls, LHS, RHS, elementbytes)
    elseif RHS.head === :call
        f = first(RHS.args)
        if f === :getindex
            add_load_getindex!(ls, LHS, RHS, elementbytes)
        elseif f === :zero || f === :one
            c = gensym(:constant)
            pushpreamble!(ls, Expr(:(=), c, RHS))
            add_constant!(ls, c, [keys(ls.loops)...], LHS, f, elementbytes)
        else
            add_compute!(ls, LHS, RHS, elementbytes)
        end
    else
        throw("Expression not recognized:\n$x")
    end
end
add_operation!(ls::LoopSet, RHS::Expr, elementbytes::Int = 8) = add_operation!(ls, gensym(:LHS), RHS, elementbytes)
function add_operation!(
    ls::LoopSet, LHS_sym::Symbol, RHS::Expr, LHS_ref::ArrayReferenceMetaPosition, elementbytes::Int = 8
)
    if RHS.head === :ref# || (RHS.head === :call && first(RHS.args) === :getindex)
        add_load!(ls, LHS_sym, LHS_ref, elementbytes)
    elseif RHS.head === :call
        f = first(RHS.args)
        if f === :getindex
            add_load!(ls, LHS_sym, LHS_ref, elementbytes)
        elseif f === :zero || f === :one
            c = gensym(:constant)
            pushpreamble!(ls, Expr(:(=), c, RHS))
            add_constant!(ls, c, [keys(ls.loops)...], LHS_sym, f, elementbytes)
        else
            add_compute!(ls, LHS_sym, RHS, elementbytes, LHS_ref)
        end
    else
        throw("Expression not recognized:\n$x")
    end
end
numsym(x)::Symbol = iszero(x) ? :zero : (isone(x) ? :one : :numconst )
function Base.push!(ls::LoopSet, ex::Expr, elementbytes::Int = 8)
    if ex.head === :call
        finex = first(ex.args)::Symbol
        if finex === :setindex!
            add_store_setindex!(ls, ex, elementbytes)
        else
            throw("Function $finex not recognized.")
        end
    elseif ex.head === :(=)
        LHS = ex.args[1]
        RHS = ex.args[2]
        if LHS isa Symbol
            if RHS isa Expr
                add_operation!(ls, LHS, RHS, elementbytes)
            else
                deps = [keys(ls.loops)...]
                if RHS isa Number
                    instr = if RHS isa Float64 # is this easier on the compiler?
                        numsym(RHS)
                    elseif RHS isa Int
                        numsym(RHS)
                    elseif RHS isa Float32
                        numsym(RHS)
                    else RHS isa Number
                        numsym(RHS)
                    end
                    add_constant!(ls, RHS, deps, LHS, instr, elementbytes)
                elseif RHS isa Symbol
                    add_constant!(ls, RHS, deps, LHS, :constsym, elementbytes)
                else
                    add_constant!(ls, RHS, deps, LHS, :constmisc, elementbytes)
                end
            end
        elseif LHS isa Expr
            @assert LHS.head === :ref
            if RHS isa Symbol
                add_store_ref!(ls, RHS, LHS, elementbytes)
            elseif RHS isa Expr
                # need to check if LHS appears in RHS
                # assign RHS to lrhs
                array, rawindices = ref_from_expr(LHS)
                mpref = array_reference_meta!(ls, array, rawindices, elementbytes)
                ref = mpref.mref.ref
                id = findfirst(r -> r == ref, ls.refs_aliasing_syms)
                lrhs = id === nothing ? gensym(:RHS) : ls.syms_aliasing_refs[id]
                add_operation!(ls, lrhs, RHS, mpref, elementbytes)
                add_store!( ls, lrhs, mpref, elementbytes )
            end
        else
            throw("LHS not understood:\n$LHS")
        end
    elseif ex.head === :block
        add_block!(ls, ex)
    elseif ex.head === :for
        add_loop!(ls, ex)
    else
        throw("Don't know how to handle expression:\n$ex")
    end
end

function addoptoorder!(
    lo::LoopOrder, included_vars::Vector{Bool}, place_after_loop::Vector{Bool}, op::Operation, loopsym::Symbol, _n::Int, unrolled::Symbol, tiled::Symbol, loopistiled::Bool
)
    id = identifier(op)
    included_vars[id] && return nothing
    loopsym ∈ loopdependencies(op) || return nothing
    included_vars[id] = true
    for opp ∈ parents(op) # ensure parents are added first
        addoptoorder!(lo, included_vars, place_after_loop, opp, loopsym, _n, unrolled, tiled, loopistiled)
    end
    isunrolled = (unrolled ∈ loopdependencies(op)) + 1
    istiled = (loopistiled ? (tiled ∈ loopdependencies(op)) : false) + 1
    # optype = Int(op.node_type) + 1
    after_loop = place_after_loop[id] + 1
    push!(lo[isunrolled,istiled,after_loop,_n], op)
    set_upstream_family!(place_after_loop, op, false) # parents that have already been included are not moved, so no need to check included_vars to filter
    nothing
end

function fillorder!(ls::LoopSet, order::Vector{Symbol}, loopistiled::Bool)
    lo = ls.loop_order
    ro = lo.loopnames # reverse order; will have same order as lo
    nloops = length(order)
    if loopistiled
        tiled    = order[1]
        unrolled = order[2]
    else
        tiled = Symbol("##UNDEFINED##")
        unrolled = first(order)
    end
    ops = operations(ls)
    nops = length(ops)
    included_vars = resize!(ls.included_vars, nops)
    fill!(included_vars, false)
    place_after_loop = resize!(ls.place_after_loop, nops)
    fill!(ls.place_after_loop, true)
    # to go inside out, we just have to include all those not-yet included depending on the current sym
    empty!(lo)
    for _n ∈ 1:nloops
        n = 1 + nloops - _n
        ro[_n] = loopsym = order[n]
        #loopsym = order[n]
        for op ∈ ops
            addoptoorder!( lo, included_vars, place_after_loop, op, loopsym, _n, unrolled, tiled, loopistiled )
        end
    end
end

# function define_remaining_ops!(
#     ls::LoopSet, vectorized::Symbol, W, unrolled, tiled, U::Int
# )
#     ops = operations(ls)
#     for (id,incl) ∈ enumerate(ls.included_vars)
#         if !incl
#             op = ops[id]
#             length(reduceddependencies(op)) == 0 && lower!( ls.preamble, op, vectorized, W, unrolled, tiled, U, nothing, nothing )
#         end
#     end
# end

# function depends_on_assigned(op::Operation, assigned::Vector{Bool})
#     for p ∈ parents(op)
#         p === op && continue # don't fall into recursive loop when we have updates, eg a = a + b
#         assigned[identifier(op)] && return true
#         depends_on_assigned(p, assigned) && return true
#     end
#     false
# end
# ind gets increased across tiles / unroll, so we need steps.
# function replace_ind_in_offset!(offset::Vector, op::Operation, ind::Int, t)
#     t == 0 && return nothing
#     var = op.variable
#     siter = op.symbolic_metadata[ind]
#     striden = op.numerical_metadata[ind]
#     strides = Symbol(:stride_, var)
#     offset[ind] = if tstriden == -1
#         Expr(:call, :*, Expr(:call, :+, strides, t), siter)
#     else
#         Expr(:call, :*, striden + t, siter)
#     end
#     nothing
# end





