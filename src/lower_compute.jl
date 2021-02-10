

function load_constrained(op, u₁loop, u₂loop, innermost_loop, forprefetch = false)
    loopdeps = loopdependencies(op)
    dependsonu₁ = u₁loop ∈ loopdeps
    dependsonu₂ = u₂loop ∈ loopdeps
    if forprefetch
        (dependsonu₁ & dependsonu₂) || return false
    end
    unrolleddeps = Symbol[]
    dependsonu₁ && push!(unrolleddeps, u₁loop)
    dependsonu₂ && push!(unrolleddeps, u₂loop)
    forprefetch && push!(unrolleddeps, innermost_loop)
    any(parents(op)) do opp
        isload(opp) && all(in(loopdependencies(opp)), unrolleddeps)
    end
end
function check_if_remfirst(ls, ua)
    usorig = ls.unrollspecification[]
    @unpack u₁, u₁loopsym, u₂loopsym, u₂max = ua
    u₁loop = getloop(ls, u₁loopsym)
    u₂loop = getloop(ls, u₂loopsym)
    if isstaticloop(u₁loop) && (usorig.u₁ != u₁)
        return true
    end
    if isstaticloop(u₂loop) && (usorig.u₂ != u₂max)
        return true
    end
    false
end
function sub_fmas(ls::LoopSet, op::Operation, ua::UnrollArgs)
    @unpack u₁, u₁loopsym, u₂loopsym, u₂max = ua
    !(load_constrained(op, u₁loopsym, u₂loopsym, first(names(ls))) || check_if_remfirst(ls, ua))
end

struct FalseCollection end
Base.getindex(::FalseCollection, i...) = false
function parent_unroll_status(op::Operation, u₁loop::Symbol, u₂loop::Symbol, ::Nothing)
    # map(opp -> isunrolled_sym(opp, u₁loop), parents(op)), map(opp -> isunrolled_sym(opp, u₂loop), parents(op))
    map(opp -> isunrolled_sym(opp, u₁loop), parents(op)), FalseCollection()
end
# function parent_unroll_status(op::Operation, u₁loop::Symbol, u₂loop::Symbol, ::Nothing)
#     vparents = parents(op);
#     parent_names = Vector{Symbol}(undef, length(vparents))
#     parents_u₁syms = Vector{Bool}(undef, length(vparents))
#     # parents_u₂syms = Vector{Bool}(undef, length(vparents))
#     for i ∈ eachindex(vparents)
#         parent_names[i], parents_u₁syms[i], _ = variable_name_and_unrolled(vparents[i], u₁loop, u₂loop, nothing)
#     end
#     parent_names, parents_u₁syms, FalseCollection()
# end
function parent_unroll_status(op::Operation, u₁loop::Symbol, u₂loop::Symbol, u₂iter::Int)
    vparents = parents(op);
    # parent_names = Vector{Symbol}(undef, length(vparents))
    parents_u₁syms = Vector{Bool}(undef, length(vparents))
    parents_u₂syms = Vector{Bool}(undef, length(vparents))
    for i ∈ eachindex(vparents)
        # parent_names[i], parents_u₁syms[i], parents_u₂syms[i] = variable_name_and_unrolled(vparents[i], u₁loop, u₂loop, u₂iter)
        parents_u₁syms[i], parents_u₂syms[i] = isunrolled_sym(vparents[i], u₁loop, u₂loop, u₂iter)
    end
    # parent_names, parents_u₁syms, parents_u₂syms
    parents_u₁syms, parents_u₂syms
end

# """
#     Requires a parents_op argument, because it may `===` parents(op), due to previous divergence, e.g. to handle unrolling.
# """
# function isreducingidentity!(q::Expr, op::Operation, parents_op::Vector{Operation}, U::Int, u₁loop::Symbol, u₂loop::Symbol, vectorized::Symbol, suffix)
#     vparents = copy(parents_op) # don't mutate the original!
#     for (i,opp) ∈ enumerate(parents_op)
#         @show opp vectorized ∈ loopdependencies(opp), vectorized ∈ reducedchildren(opp) # must reduce
#         @show vectorized, loopdependencies(opp), reducedchildren(opp) # must reduce
#         if vectorized ∈ loopdependencies(opp) || vectorized ∈ reducedchildren(opp) # must reduce
#             loopdeps = [l for l ∈ loopdependencies(opp) if l !== vectorized]
#             @show opp
#             reductinstruct = reduction_to_scalar(instruction(opp))
            
#             reducedparent = Operation(
#                 opp.identifier, gensym(opp.variable), opp.elementbytes, Instruction(:LoopVectorization, reductinstruct), opp.node_type,
#                 loopdeps, opp.reduced_deps, opp.parents, opp.ref, opp.reduced_children
#             )
#             pname,   pu₁,  pu₂ = variable_name_and_unrolled(opp, u₁loop, u₂loop, suffix)
#             rpname, rpu₁, rpu₂ = variable_name_and_unrolled(reducedparent, u₁loop, u₂loop, suffix)
#             @assert pu₁ == rpu₁ && pu₂ == rpu₂
#             if rpu₁
#                 for u ∈ 0:U-1
#                     push!(q.args, Expr(:(=), Symbol(rpname,u), Expr(:call, lv(reductinstruct), Symbol(pname,u))))
#                 end
#             else
#                 push!(q.args, Expr(:(=), rpname, Expr(:call, lv(reductinstruct), pname)))
#             end
#             vparents[i] = reducedparent
#         end
#     end
#     vparents
# end

function _add_loopvalue!(ex::Expr, loopval::Symbol, vectorized::Symbol, u::Int)
    if loopval === vectorized
        if iszero(u)
            push!(ex.args, _MMind(Expr(:call, lv(:staticp1), loopval)))
        elseif isone(u)
            push!(ex.args, Expr(:call, lv(:vadd_fast), VECTORWIDTHSYMBOL, _MMind(Expr(:call, lv(:staticp1), loopval))))
        else
            push!(ex.args, Expr(:call, lv(:vadd_fast), Expr(:call, lv(:vmul_fast), VECTORWIDTHSYMBOL, u), _MMind(Expr(:call, lv(:staticp1), loopval))))
        end
    else
        push!(ex.args, Expr(:call, lv(:vadd_fast), loopval, staticexpr(u+1)))
    end
end
function add_loopvalue!(instrcall::Expr, loopval, ua::UnrollArgs, u₁::Int)
    @unpack u₁loopsym, u₂loopsym, vectorized, suffix = ua
    if loopval === u₁loopsym #parentsunrolled[n]
        if isone(u₁)
            _add_loopvalue!(instrcall, loopval, vectorized, 0)
        else
            t = Expr(:tuple)
            for u ∈ 0:u₁-1
                _add_loopvalue!(t, loopval, vectorized, u)
            end
            push!(instrcall.args, Expr(:call, lv(:VecUnroll), t))
        end
    elseif !isnothing(suffix) && suffix > 0 && loopval === u₂loopsym
        _add_loopvalue!(instrcall, loopval, vectorized, suffix)
    elseif loopval === vectorized
        push!(instrcall.args, _MMind(Expr(:call, lv(:staticp1), loopval)))
    else
        push!(instrcall.args, Expr(:call, lv(:staticp1), loopval))
    end
end

vecunrolllen(::Type{VecUnroll{N,W,T,V}}) where {N,W,T,V} = (N::Int + 1)
vecunrolllen(_) = -1
function ifelselastexpr(hasf::Bool, M::Int, vargtypes, K::Int)
    t = Expr(:tuple)
    q = Expr(:block, Expr(:meta,:inline))
    vargs = map(k -> Symbol(:varg_,k), 1:K)
    lengths = Vector{Int}(undef, K);
    for k ∈ 1:K
        lengths[k] = l = vecunrolllen(vargtypes[k])
        if hasf
            if l == -1
                push!(q.args, :($(vargs[k]) = getfield(vargs, $k, false)))
            else
                push!(q.args, :($(vargs[k]) = data(getfield(vargs, $k, false))))
            end
        elseif l != -1
            varg = vargs[k]
            vargs[k] = dvarg = Symbol(:d, varg)
            push!(q.args, :($dvarg = data($varg)))
        end
    end
    N = last(lengths)
    # @show N, M lengths vargs
    start = hasf ? 1 : M
    for m ∈ 1:start-1
        push!(t.args, :(getfield($(vargs[K]), $m, false)))
    end
    for m ∈ start:M
        call = if hasf
            m == M ? Expr(:call, :ifelse, :f, :m) : Expr(:call, :f)
        else# m == M because !hasf
            Expr(:call, :ifelse, :m)
        end
        for k ∈ 1:K
            if lengths[k] == -1
                push!(call.args, vargs[k])
            else
                # @assert (k == K) || (lengths[k] == M)
                push!(call.args, :(getfield($(vargs[k]), $m, false)))
            end
        end
        if N == -1
            push!(q.args, call)
            return q
        end
        push!(t.args, call)
    end
    for m ∈ M+1:N
        push!(t.args, :(getfield($(vargs[K]), $m, false)))
    end
    # push!(q.args, :(VecUnroll($t)::VecUnroll{$N,$W,$T,$V}))
    push!(q.args, :(VecUnroll($t)))
    q
end
@generated function ifelselast(f::F, m::Mask{W}, ::StaticInt{M}, vargs::Vararg{Any,K}) where {F,W,K,M}
    # 1+1
    # @show vargs K
    ifelselastexpr(true, M, vargs, K)
end
@generated function ifelselast(m::Mask{W}, ::StaticInt{M}, varg_1::V1, varg_2::V2) where {W,V1,V2,M}
    1+1
    # @show V1 V2 W
    ifelselastexpr(false, M, (V1,V2), 2)
end
@generated function partialmap(f::F, default::D, ::StaticInt{M}, vargs::Vararg{Any,K}) where {F,M,K,D}
    lengths = Vector{Int}(undef, K);
    q = Expr(:block, Expr(:meta,:inline))
    syms = Vector{Symbol}(undef, K)
    for k ∈ 1:K
        lengths[k] = l = vecunrolllen(vargs[k])
        @assert (l == -1) || (l ≥ M)
        syms[k] = symk = Symbol(:vargs_,k)
        extractq = :(getfield(vargs, $k, false))
        if l != -1
            extractq = :(data($extractq))
        end
        push!(q.args, :($symk = $extractq))
    end
    N = maximum(lengths)
    Dlen = vecunrolllen(D)
    @assert N == Dlen
    if Dlen == -1
        @assert M == 1
    else
        push!(q.args, :(dd = data(default)))
    end
    t = Expr(:tuple)
    for m ∈ 1:M
        call = Expr(:call, :f)
        for k ∈ 1:K
            if lengths[k] == -1
                push!(call.args, syms[k])
            else
                push!(call.args, Expr(:call, :getfield, syms[k], m, false))
            end
        end
        if N == -1
            push!(q.args, call)
            return q
        end
        push!(t.args, call)
    end
    for m ∈ M+1:N 
        push!(t.args, :(getfield(dd, $m, false)))
    end
    push!(q.args, :(VecUnroll($t)))
    q
end

function parent_op_name(parents_op, n, modsuffix, suffix_, parents_u₁syms, parents_u₂syms, u₁, opisvectorized, tiledouterreduction)
    opp = parents_op[n]
    parent = mangledvar(opp)
    if n == tiledouterreduction
        parent = Symbol(parent, modsuffix)
    else
        if parents_u₂syms[n]
            parent = Symbol(parent, suffix_)
        end
    end
    parent = Symbol(parent, '_', parents_u₁syms[n] ? u₁ : 1)
    if opisvectorized && isload(opp) && (!isvectorized(opp))
        # && n != tiledouterreduction && !(parents_u₁syms[n] & parents_u₂syms[n])
        parent = Symbol(parent, "##broadcasted##")
    end
    parent
end

function lower_compute!(
    q::Expr, op::Operation, ls::LoopSet, ua::UnrollArgs, mask::Union{Nothing,Symbol,Unsigned} = nothing,
)
    @unpack u₁, u₁loopsym, u₂loopsym, vectorized, suffix = ua
    var = name(op)
    instr = instruction(op)
    parents_op = parents(op)
    nparents = length(parents_op)
    mvar, u₁unrolledsym, u₂unrolledsym = variable_name_and_unrolled(op, u₁loopsym, u₂loopsym, suffix)
    # @show mvar, mangledvar(op), suffix
    opunrolled = u₁unrolledsym || u₁loopsym ∈ loopdependencies(op)
    # parent_names, parents_u₁syms, parents_u₂syms = parent_unroll_status(op, u₁loop, u₂loop, suffix)
    parents_u₁syms, parents_u₂syms = parent_unroll_status(op, u₁loopsym, u₂loopsym, suffix)
    tiledouterreduction = if isnothing(suffix)
        suffix_ = nothing
        -1
    else
        suffix_ = Symbol(suffix, :_)
        isouterreduction(op)
    end
    if !opunrolled && any(parents_u₁syms) # TODO: Clean up this mess, refactor the naming code, putting it in one place and have everywhere else use it for easy equivalence.
        parents_op = copy(parents_op) # don't mutate the original!
        for i ∈ eachindex(parents_u₁syms)
            parents_u₁syms[i] || continue
            parents_u₁syms[i] = false
            parentop = parents_op[i]
            i == tiledouterreduction && isconstant(parentop) && continue
            newparentop = Operation(
                parentop.identifier, gensym(parentop.variable), parentop.elementbytes, parentop.instruction, parentop.node_type,
                parentop.dependencies, parentop.reduced_deps, parentop.parents, parentop.ref, parentop.reduced_children
            )
            parentname = mangledvar(parentop)
            newparentname = mangledvar(newparentop)
            parents_op[i] = newparentop
            if parents_u₂syms[i]
                parentname = Symbol(parentname, suffix_)
                newparentname = Symbol(newparentname, suffix_)
            end
            if isconstant(newparentop)
                push!(q.args, Expr(:(=), Symbol(newparentname, '_', 1), Symbol(parentname, '_', 1)))
            else
                newpname = Symbol(newparentname, '_', u₁)
                push!(q.args, Expr(:(=), newpname, Symbol(parentname, '_', u₁)))
                # @show newparentop op instruction(newparentop)
                reduce_expr!(q, newparentname, instruction(newparentop), u₁, -1)
                push!(q.args, Expr(:(=), Symbol(newparentname, '_', 1), Symbol(newparentname, "##onevec##")))
            end
        end
    end
    # if suffix === nothing# &&
    # end
    # if instr.instr === :div_fast
    #     @show op, suffix, parents_u₂syms parents(op)
    #     @show isu₂unrolled.(parents(op))
    # end
    # cache unroll and tiling check of parents
    # not broadcasted, because we use frequent checks of individual bools
    # making BitArrays inefficient.
    # parentsyms = [opp.variable for opp ∈ parents(op)]
    Uiter = opunrolled ? u₁ - 1 : 0
    # @show mvar, opunrolled, u₁, u₁loopsym, u₂loopsym
    isreduct = isreduction(op)
    if Base.libllvm_version < v"11.0.0" && !isnothing(suffix) && isreduct# && (iszero(suffix) || (ls.unrollspecification[].u₂ - 1 == suffix))
        # instrfid = findfirst(isequal(instr.instr), (:vfmadd, :vfnmadd, :vfmsub, :vfnmsub))
        instrfid = findfirst(isequal(instr.instr), (:vfmadd_fast, :vfnmadd_fast, :vfmsub_fast, :vfnmsub_fast))
        # want to instcombine when parent load's deps are superset
        # also make sure opp is unrolled
        if !isnothing(instrfid) && (opunrolled && u₁ > 1) && sub_fmas(ls, op, ua)
            specific_fmas = Base.libllvm_version >= v"11.0.0" ? (:vfmadd, :vfnmadd, :vfmsub, :vfnmsub) : (:vfmadd231, :vfnmadd231, :vfmsub231, :vfnmsub231)
            # specific_fmas = (:vfmadd231, :vfnmadd231, :vfmsub231, :vfnmsub231)
            instr = Instruction(specific_fmas[instrfid])
        end
    end
    reduceddeps = reduceddependencies(op)
    vecinreduceddeps = isreduct && vectorized ∈ reduceddeps
    maskreduct = !isnothing(mask) && vecinreduceddeps #any(opp -> opp.variable === var, parents_op)
    # if vecinreduceddeps && vectorized ∉ loopdependencies(op) # screen parent opps for those needing a reduction to scalar
    #     # parents_op = reduce_vectorized_parents!(q, op, parents_op, U, u₁loopsym, u₂loopsym, vectorized, suffix)
    #     isreducingidentity!(q, op, parents_op, U, u₁loopsym, u₂loopsym, vectorized, suffix) && return
    # end    
    # if a parent is not unrolled, the compiler should handle broadcasting CSE.
    # because unrolled/tiled parents result in an unrolled/tiled dependendency,
    # we handle both the tiled and untiled case here.
    # bajillion branches that go the same way on each iteration
    # but smaller function is probably worthwhile. Compiler could theoreically split anyway
    # but I suspect that the branches are so cheap compared to the cost of everything else going on
    # that smaller size is more advantageous.
    opisvectorized = isvectorized(op)
    modsuffix = 0
    # for u ∈ 0:Uiter
    isouterreduct = false
    instrcall = callexpr(instr)
    varsym = if tiledouterreduction > 0 # then suffix !== nothing
        # modsuffix = ((u + suffix*(Uiter + 1)) & 7)
        isouterreduct = true
        modsuffix = suffix % tiled_outerreduct_unroll(ls)
        Symbol(mangledvar(op), modsuffix)
        # Symbol(mvar, modsuffix)
        # elseif u₁unrolledsym
        #     Symbol(mvar, u)
    elseif isanouterreduction(ls, op)
        isouterreduct = true
        Ureduct = ureduct(ls)
        ufull = if Ureduct == -1 # no reducing
            ls.unrollspecification[].u₁
        else
            Ureduct
        end
        Symbol(mvar, '_', max(u₁, ufull))
    else
        Symbol(mvar, '_', ifelse(opunrolled, u₁, 1))
    end
    selfopname = varsym
    # @show op, tiledouterreduction, isouterreduct
    # if name(op) === Symbol("##op#5631")
    #     @show name(op), parents(op), name.(parents(op))
    #     parent_name = parent_op_name(parents_op, 1, modsuffix, suffix_, parents_u₁syms, parents_u₂syms, u₁, opisvectorized, tiledouterreduction)
    #     @show parent_name
    # end
    # @show selfopname, varsym, mvar, mangledvar(op)
    selfdep = false
    # showexpr = false
    for n ∈ 1:nparents
        opp = parents_op[n]
        if isloopvalue(opp)
            loopval = first(loopdependencies(opp))
            add_loopvalue!(instrcall, loopval, ua, u₁)
        elseif name(opp) === name(op)
            selfdep = true
            if ((isvectorized(first(parents_op)) && !isvectorized(op)) && !dependent_outer_reducts(ls, op))
                parent = parent_op_name(parents_op, n, modsuffix, suffix_, parents_u₁syms, parents_u₂syms, u₁, opisvectorized, tiledouterreduction)
                selfopname = parent
                push!(instrcall.args, parent)
            else
                # @show name(parents_op[n]), name(op), mangledvar(parents_op[n]), mangledvar(op)
                push!(instrcall.args, varsym)
            end
        elseif ((!isu₂unrolled(op)) & isu₂unrolled(opp)) && (isouterreduction(opp) != -1)
            # this checks if the parent is u₂ unrolled but this operation is not, in which case we need to reduce it.
            push!(instrcall.args, reduce_expr_u₂(mangledvar(opp), instruction(opp), ureduct(ls)))
        else
            parent = parent_op_name(parents_op, n, modsuffix, suffix_, parents_u₁syms, parents_u₂syms, u₁, opisvectorized, tiledouterreduction)
            push!(instrcall.args, parent)
        end
    end
    if maskreduct
        ifelsefunc = if ls.unrollspecification[].u₁ == 1 #u₁loopsym !== vectorized
            :ifelse
        else# mask last u₁
            :ifelselast
        end
        if last(instrcall.args) == varsym
            pushfirst!(instrcall.args, lv(ifelsefunc))
            # showexpr = true
            insert!(instrcall.args, 3, mask)
            ifelsefunc === :ifelselast && insert!(instrcall.args, 4, staticexpr(u₁))
        elseif all(in(loopdependencies(op)), reduceddeps) || any(opp -> mangledvar(opp) === mangledvar(op), parents_op)
            if ifelsefunc === :ifelse
                push!(q.args, Expr(:(=), varsym, Expr(:call, lv(ifelsefunc), mask, instrcall, selfopname)))
            else
                push!(q.args, Expr(:(=), varsym, Expr(:call, lv(ifelsefunc), mask, staticexpr(u₁), instrcall, selfopname)))
            end
            return
        elseif selfdep
            # @show op, isouterreduct, maskreduct, instr
            pushfirst!(instrcall.args, lv(:partialmap))
            insert!(instrcall.args, 3, selfopname)
            insert!(instrcall.args, 4, staticexpr(u₁))
        end
    elseif isouterreduct && selfdep && (opunrolled) && (u₁ < ls.unrollspecification[].u₁)
        # @show op, isouterreduct, maskreduct, instr
        # needed for cases like `myselfdotavx(A')`, where we have an unrolled reduction
        # 
        # names could disagree
        pushfirst!(instrcall.args, lv(:partialmap))
        insert!(instrcall.args, 3, selfopname)
        insert!(instrcall.args, 4, staticexpr(u₁))
    end
    # if showexpr
    #     for i ∈ eachindex(parents_op)
    #         push!(q.args, :(@show $(instrcall.args[end+1-i])))
    #     end
    #     instrcall = :(@show $instrcall)
    # end
    if instr.instr === :identity && isone(length(parents_op))
        push!(q.args, Expr(:(=), varsym, instrcall.args[2]))
    elseif identifier(op) ∉ ls.outer_reductions && should_broadcast_op(op)
        push!(q.args, Expr(:(=), varsym, Expr(:call, lv(:vbroadcast), VECTORWIDTHSYMBOL, instrcall)))
    else
        push!(q.args, Expr(:(=), varsym, instrcall))
    end
    # end
end


