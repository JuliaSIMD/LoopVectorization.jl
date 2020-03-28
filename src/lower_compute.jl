# A compute op needs to know the unrolling and tiling status of each of its parents.

struct FalseCollection end
Base.getindex(::FalseCollection, i...) = false
function lower_compute!(
    q::Expr, op::Operation, vectorized::Symbol, W::Symbol, unrolled::Symbol, tiled::Symbol, U::Int,
    suffix::Union{Nothing,Int}, mask::Union{Nothing,Symbol,Unsigned} = nothing,
    opunrolled = unrolled ∈ loopdependencies(op)
)

    var = name(op)
    instr = instruction(op)
    parents_op = parents(op)
    mvar = mangledvar(op)
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
    parentsunrolled = isunrolled_sym.(parents_op, unrolled, tiled)
    if instr.instr === :identity && name(first(parents_op)) === var && isone(length(parents_op))
        if (opunrolled == first(parentsunrolled)) && ((!isnothing(suffix)) == first(parentstiled))
            return
        end
    end
    unrollsym = isunrolled_sym(op, unrolled, suffix)
    if !opunrolled && any(parentsunrolled)
        parents_op = copy(parents_op)
        for i ∈ eachindex(parentsunrolled)
            parentsunrolled[i] || continue
            parentsunrolled[i] = false
            parentop = parents_op[i]
            # @show op, parentop
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
            if isconstant(newparentop)
                push!(q.args, Expr(:(=), newparentname, Symbol(parentname, 0)))
                continue
            else
                for u ∈ 0:U-1
                    push!(q.args, Expr(:(=), Symbol(newparentname, u), Symbol(parentname, u)))
                end
                reduce_expr!(q, newparentname, Instruction(reduction_to_single_vector(instruction(newparentop))), U)
                push!(q.args, Expr(:(=), newparentname, Symbol(newparentname, 0)))
            end
        end
    end
    # cache unroll and tiling check of parents
    # not broadcasted, because we use frequent checks of individual bools
    # making BitArrays inefficient.
    # parentsyms = [opp.variable for opp ∈ parents(op)]
    Uiter = opunrolled ? U - 1 : 0
    isreduct = isreduction(op)
    # if instr.instr === :vfmadd_fast
        # diffdeps = !any(opp -> isload(opp) && all(in(loopdependencies(opp)), loopdependencies(op)), parents(op)) # want to instcombine when parent load's deps are superset
        # @show suffix, !isnothing(suffix), isreduct, diffdeps
    # end
    if !isnothing(suffix) && isreduct
        # instrfid = findfirst(isequal(instr.instr), (:vfmadd, :vfnmadd, :vfmsub, :vfnmsub))
        instrfid = findfirst(isequal(instr.instr), (:vfmadd_fast, :vfnmadd_fast, :vfmsub_fast, :vfnmsub_fast))
        if instrfid !== nothing && !any(opp -> isload(opp) && all(in(loopdependencies(opp)), loopdependencies(op)), parents(op)) # want to instcombine when parent load's deps are superset
            instr = Instruction((:vfmadd231, :vfnmadd231, :vfmsub231, :vfnmsub231)[instrfid])
        end
    end
    # @show instr.instr
    maskreduct = mask !== nothing && isreduct && vectorized ∈ reduceddependencies(op) #any(opp -> opp.variable === var, parents_op)
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
            # modsuffix = ((u + suffix*U) & 3)
            modsuffix = (suffix & 3)
            Symbol(mvar, modsuffix)
        elseif unrollsym
            Symbol(mvar, u)
        else
            mvar
        end
        for n ∈ 1:nparents
            if isloopvalue(parents_op[n])
                loopvalue = first(loopdependencies(parents_op[n]))
                if u > 0 && loopvalue === unrolled #parentsunrolled[n]
                    if loopvalue === vectorized
                        push!(instrcall.args, Expr(:call, :+, loopvalue, Expr(:call, lv(:valmul), W, u)))
                    else
                        push!(instrcall.args, Expr(:call, :+, loopvalue, u))
                    end
                else
                    push!(instrcall.args, loopvalue)
                end
            else
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
        end
        if maskreduct && (u == Uiter || unrolled !== vectorized) # only mask last
            if last(instrcall.args) == varsym
                pushfirst!(instrcall.args, lv(:vifelse))
                insert!(instrcall.args, 3, mask)
            else
                push!(q.args, Expr(:(=), varsym, Expr(:call, lv(:vifelse), mask, instrcall, varsym)))
                continue
            end
        end
        if instr.instr === :identity && isone(length(parents_op))
            push!(q.args, Expr(:(=), varsym, instrcall.args[2]))
        else
            push!(q.args, Expr(:(=), varsym, instrcall))
        end
    end
end


