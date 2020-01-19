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
function update_deps!(deps::Vector{Symbol}, reduceddeps::Vector{Symbol}, parent::Operation)
    mergesetdiffv!(deps, loopdependencies(parent), reduceddependencies(parent))
    if !(isload(parent) || isconstant(parent)) && parent.instruction.instr ∉ (:reduced_add, :reduced_prod, :reduce_to_add, :reduce_to_prod)
        mergesetv!(reduceddeps, reduceddependencies(parent))
    end
    nothing
end

function pushparent!(parents::Vector{Operation}, deps::Vector{Symbol}, reduceddeps::Vector{Symbol}, parent::Operation)
    push!(parents, parent)
    update_deps!(deps, reduceddeps, parent)
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
    # if length(reduceddependencies(reductinit)) == 0
        # setdiffv!(reduceddeps, deps, loopdependencies(reductinit))
    # else
    setdiffv!(reduceddeps, deps, loopdependencies(reductinit))
    # end
    # mergesetv!(reduceddependencies(reductinit), reduceddeps)
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

function add_compute!(
    ls::LoopSet, LHS::Symbol, instr, parents::Vector{Operation}, elementbytes
)
    deps = Symbol[]
    reduceddeps = Symbol[]
    foreach(parent -> update_deps!(deps, reduceddeps, parent), parents)
    op = Operation(length(operations(ls)), LHS, elementbytes, instr, compute, deps, reduceddeps, parents)
    pushop!(ls, op, LHS)
end

