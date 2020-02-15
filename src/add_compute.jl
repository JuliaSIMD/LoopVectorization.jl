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
# Everything in arg2 (s1) that isn't in arg3 (s2) is added to arg1 (s3)
function setdiffv!(s3::AbstractVector{T}, s1::AbstractVector{T}, s2::AbstractVector{T}) where {T}
    for s ∈ s1
        (s ∈ s2) || (s ∉ s3 && push!(s3, s))
    end
end
function setdiffv!(s4::AbstractVector{T}, s3::AbstractVector{T}, s1::AbstractVector{T}, s2::AbstractVector{T}) where {T}
    for s ∈ s1
        (s ∈ s2) ? (s ∉ s4 && push!(s4, s)) : (s ∉ s3 && push!(s3, s))
    end
end
function update_deps!(deps::Vector{Symbol}, reduceddeps::Vector{Symbol}, parent::Operation)
    mergesetv!(deps, loopdependencies(parent))#, reduceddependencies(parent))        
    if !(isload(parent) || isconstant(parent)) && !isreductcombineinstr(parent)
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
    parents::Vector{Operation}, deps::Vector{Symbol}, reduceddeps::Vector{Symbol}, ls::LoopSet, var, elementbytes::Int, position::Int
)
    parent = if var isa Symbol
        getop(ls, var, elementbytes)
    elseif var isa Expr #CSE candidate
        add_operation!(ls, gensym(:temporary), var, elementbytes, position)
    else # assumed constant
        add_constant!(ls, var, elementbytes)
        # add_constant!(ls, var, deps, gensym(:loopredefconst), elementbytes)
    end
    pushparent!(parents, deps, reduceddeps, parent)
end
function add_reduction!(
    parents::Vector{Operation}, deps::Vector{Symbol}, reduceddeps::Vector{Symbol}, ls::LoopSet, var::Symbol, elementbytes::Int
)
    get!(ls.opdict, var) do
        add_constant!(ls, var, elementbytes)
    end
end
function search_tree(opv::Vector{Operation}, var::Symbol) # relies on cycles being forbidden
    for opp ∈ opv
        name(opp) === var && return true
        search_tree(parents(opp), var) && return true
    end
    false
end
function update_reduction_status!(parentvec::Vector{Operation}, deps::Vector{Symbol}, parent::Symbol)
    for opp ∈ parentvec
        if name(opp) === parent
            mergesetv!(reducedchildren(opp), deps)
            break
        elseif search_tree(parents(opp), parent)
            mergesetv!(reducedchildren(opp), deps)
            update_reduction_status!(parents(opp), deps, parent)
            break
        end
    end
end
# function add_compute!(ls::LoopSet, op::Operation)
    # @assert iscompute(op)
    # pushop!(ls, child, name(op))
# end
function isreductzero(op::Operation, ls::LoopSet, reduct_zero::Symbol)
    isconstant(op) || return false
    reduct_zero === op.instruction.mod && return true
    if reduct_zero === :zero
        iszero(ls, op) && return true
    elseif reduct_zero === :one
        isone(ls, op) && return true
    end
    false
end

function add_reduction_update_parent!(
    vparents::Vector{Operation}, deps::Vector{Symbol}, reduceddeps::Vector{Symbol}, ls::LoopSet,
    parent::Operation, instr::Symbol, directdependency::Bool, elementbytes::Int
)
    var = name(parent)
    isouterreduction = parent.instruction === LOOPCONSTANT
    Instr = instruction(ls, instr)
    instrclass = reduction_instruction_class(Instr) # key allows for faster lookups
    # if parent is not an outer reduction...
    if !isouterreduction
        # and parent is not a reduction_zero
        reduct_zero = reduction_zero(instrclass)
        reductcombine = reduction_scalar_combine(instrclass)
        reductsym = gensym(:reduction)
        reductinit = add_constant!(ls, gensym(:reductzero), loopdependencies(parent), reductsym, elementbytes, :numericconstant)
        if reduct_zero === :zero
            push!(ls.preamble_zeros, (identifier(reductinit), IntOrFloat))
        elseif reduct_zero === :one
            push!(ls.preamble_ones, (identifier(reductinit), IntOrFloat))
        else
            if reductzero === :true || reductzero === :false
                pushpreamble!(ls, Expr(:(=), name(reductinit), reductzero))
            else
                pushpreamble!(ls, Expr(:(=), name(reductinit), Expr(:call, reductzero, ls.T)))
            end
            pushpreamble!(ls, op, name, reductinit)
        end
        if isreductzero(parent, ls, reduct_zero)
            reductcombine = reduction_combine_to(instrclass)
        end
    else
        reductinit = parent
        reductsym = var
        reductcombine = Symbol("")
    end
    combineddeps = copy(deps); mergesetv!(combineddeps, reduceddeps)
    directdependency && pushparent!(vparents, deps, reduceddeps, reductinit)#parent) # deps and reduced deps will not be disjoint
    update_reduction_status!(vparents, reduceddeps, name(reductinit))
    # this is the op added by add_compute
    op = Operation(length(operations(ls)), reductsym, elementbytes, instr, compute, deps, reduceddeps, vparents)
    parent.instruction === LOOPCONSTANT && push!(ls.outer_reductions, identifier(op))
    opout = pushop!(ls, op, var) # note this overwrites the entry in the operations dict, but not the vector
    # isouterreduction || iszero(length(reduceddeps)) && return opout
    isouterreduction && return opout
    # create child op, which is the reduction combination
    childrdeps = Symbol[]; childparents = Operation[ op, parent ]
    childdeps = loopdependencies(reductinit)
    setdiffv!(childrdeps, loopdependencies(op), childdeps)
    child = Operation(
        length(operations(ls)), name(parent), elementbytes, reductcombine, compute, childdeps, childrdeps, childparents
    )
    pushop!(ls, child, name(parent))
    opout
end
function add_compute!(
    ls::LoopSet, var::Symbol, ex::Expr, elementbytes::Int, position::Int,
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
                    add_load!(ls, var, argref, elementbytes)
                else
                    pushparent!(parents, deps, reduceddeps, add_load!(ls, gensym(:tempload), argref, elementbytes))
                end
            else
                add_parent!(parents, deps, reduceddeps, ls, arg, elementbytes, position)
            end
        elseif arg ∈ ls.loopsymbols
            loopsymop = add_loopvalue!(ls, arg, elementbytes)
            pushparent!(parents, deps, reduceddeps, loopsymop)
        else
            add_parent!(parents, deps, reduceddeps, ls, arg, elementbytes, position)
        end
    end
    if iszero(length(deps)) && reduction
        loopnestview = view(ls.loopsymbols, 1:position) 
        append!(deps, loopnestview)
        append!(reduceddeps, loopnestview)
    else
        loopnestview = view(ls.loopsymbols, 1:position) 
        newloopdeps = Symbol[]; newreduceddeps = Symbol[];
        setdiffv!(newloopdeps, newreduceddeps, deps, loopnestview)
        mergesetv!(newreduceddeps, reduceddeps)
        deps = newloopdeps; reduceddeps = newreduceddeps
    end
    if reduction || search_tree(parents, var)
        parent = getop(ls, var, elementbytes)
        setdiffv!(reduceddeps, deps, loopdependencies(parent))
        if length(reduceddeps) == 0
            push!(parents, parent)
            op = Operation(length(operations(ls)), var, elementbytes, instruction(ls,instr), compute, deps, reduceddeps, parents)
            pushop!(ls, op, var)
        else
            add_reduction_update_parent!(parents, deps, reduceddeps, ls, parent, instr, reduction, elementbytes)
        end
    else
        op = Operation(length(operations(ls)), var, elementbytes, instruction(ls,instr), compute, deps, reduceddeps, parents)
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

