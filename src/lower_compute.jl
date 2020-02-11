# A compute op needs to know the unrolling and tiling status of each of its parents.

struct FalseCollection end
Base.getindex(::FalseCollection, i...) = false
function lower_compute!(
    q::Expr, op::Operation, vectorized::Symbol, W::Symbol, unrolled::Symbol, tiled::Symbol, U::Int,
    suffix::Union{Nothing,Int}, mask::Union{Nothing,Symbol,Unsigned} = nothing,
    opunrolled = unrolled ∈ loopdependencies(op)
)

    var = op.variable
    mvar = mangledvar(op)
    parents_op = parents(op)
    nparents = length(parents_op)
    parentstiled = if suffix === nothing
        optiled = false
        tiledouterreduction = -1
        FalseCollection()
    else
        tiledouterreduction = isouterreduction(op)
        suffix_ = Symbol(suffix, :_)
        if tiledouterreduction == -1
            mvar = Symbol(mvar, suffix_)
        end
        optiled = true
        [tiled ∈ loopdependencies(opp) for opp ∈ parents_op]
    end
    parentsunrolled = [unrolled ∈ loopdependencies(opp) || unrolled ∈ reducedchildren(opp) for opp ∈ parents_op]
    if !opunrolled && any(parentsunrolled)
        parents_op = copy(parents_op)
        for i ∈ eachindex(parentsunrolled)
            parentsunrolled[i] || continue
            parentsunrolled[i] = false
            parentop = parents_op[i]
            newparentop = Operation(
                parentop.identifier, gensym(parentop.variable), parentop.elementbytes, parentop.instruction, parentop.node_type,
                parentop.dependencies, parentop.reduced_deps, parentop.parents, parentop.ref, parentop.reduced_children
            )
            parentname = mangledvar(parentop)
            newparentname = mangledvar(newparentop)
            parents_op[i] = newparentop
            if parentstiled[i]
                parentname = Symbol(parentname, suffix_)
                newparentname = Symbol(newparentname, suffix_)
            end
            for u ∈ 0:U-1
                push!(q.args, Expr(:(=), Symbol(newparentname, u), Symbol(parentname, u)))
            end
            reduce_expr!(q, newparentname, Instruction(reduction_to_single_vector(instruction(newparentop))), U)
            push!(q.args, Expr(:(=), newparentname, Symbol(newparentname, 0)))
        end
    end
    instr = op.instruction
    # cache unroll and tiling check of parents
    # not broadcasted, because we use frequent checks of individual bools
    # making BitArrays inefficient.
    # parentsyms = [opp.variable for opp ∈ parents(op)]
    Uiter = opunrolled ? U - 1 : 0
    maskreduct = mask !== nothing && isreduction(op) && vectorized ∈ reduceddependencies(op) #any(opp -> opp.variable === var, parents_op)
    # if a parent is not unrolled, the compiler should handle broadcasting CSE.
    # because unrolled/tiled parents result in an unrolled/tiled dependendency,
    # we handle both the tiled and untiled case here.
    # bajillion branches that go the same way on each iteration
    # but smaller function is probably worthwhile. Compiler could theoreically split anyway
    # but I suspect that the branches are so cheap compared to the cost of everything else going on
    # that smaller size is more advantageous.
    modsuffix = 0
    for u ∈ 0:Uiter
        instrcall = Expr(instr) # Expr(:call, instr)
        varsym = if tiledouterreduction > 0 # then suffix !== nothing
            modsuffix = ((u + suffix*U) & 3)
            Symbol(mvar, modsuffix)
        elseif opunrolled
            Symbol(mvar, u)
        else
            mvar
        end
        for n ∈ 1:nparents
            parent = mangledvar(parents_op[n])
            if n == tiledouterreduction
                parent = Symbol(parent, modsuffix)
            else
                if parentstiled[n]
                    parent = Symbol(parent, suffix_)
                end
                if parentsunrolled[n]
                    parent = Symbol(parent, u)
                end
            end
            push!(instrcall.args, parent)
        end
        if maskreduct && (u == Uiter || unrolled !== vectorized) # only mask last
            push!(q.args, Expr(:(=), varsym, Expr(:call, lv(:vifelse), mask, instrcall, varsym)))
        else
            push!(q.args, Expr(:(=), varsym, instrcall))
        end
    end
end


