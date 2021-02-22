function storeinstr_preprend(op::Operation, vectorized::Symbol)
    # defaultstoreop = :vstore!
    # defaultstoreop = :vnoaliasstore!
    vectorized ∉ reduceddependencies(op) && return Symbol("")
    vectorized ∈ loopdependencies(op) && return Symbol("")
    # vectorized is not a loopdep, but is a reduced dep
    opp = first(parents(op))
    while vectorized ∉ loopdependencies(opp)
        oppold = opp
        for oppp ∈ parents(opp)
            if vectorized ∈ reduceddependencies(oppp)
                @assert opp !== oppp "More than one parent is a reduction over the vectorized variable."
                opp = oppp
            end
        end
        @assert opp !== oppold "Failed to find any parents "
    end
    reduction_to_scalar(reduction_instruction_class(instruction(opp)))
end

function reduce_expr_u₂(toreduct::Symbol, instr::Instruction, u₂::Int)
    t = Expr(:tuple)
    for u ∈ 0:u₂-1
        push!(t.args, Symbol(toreduct, u))
    end
    Expr(:call, lv(:reduce_tup), reduce_to_onevecunroll(instr), t)
end
function reduce_expr!(q::Expr, toreduct::Symbol, instr::Instruction, u₁::Int, u₂::Int)
    if u₂ != -1
        _toreduct = Symbol(toreduct, 0)
        push!(q.args, Expr(:(=), _toreduct, reduce_expr_u₂(toreduct, instr, u₂)))
    else
        _toreduct = Symbol(toreduct, '_', u₁)
    end
    if u₁ == 1
        push!(q.args, Expr(:(=), Symbol(toreduct, "##onevec##"), _toreduct))
    else
        push!(q.args, Expr(:(=), Symbol(toreduct, "##onevec##"), Expr(:call, lv(reduction_to_single_vector(instr)), _toreduct)))
        # push!(q.args, :(@show $_toreduct))
        # push!(q.args, Expr(:(=), Symbol(toreduct, "##onevec##"), :(@show $(Expr(:call, lv(reduction_to_single_vector(instr)), _toreduct)))))
    end
    nothing
end

function lower_store!(
    q::Expr, ls::LoopSet, op::Operation, ua::UnrollArgs, mask::Bool,
    reductfunc::Symbol = storeinstr_preprend(op, ua.vectorized), inds_calc_by_ptr_offset = indices_calculated_by_pointer_offsets(ls, op.ref)
)
    @unpack u₁, u₁loopsym, u₂loopsym, vectorized, u₂max, suffix = ua
    isunrolled₁ = isu₁unrolled(op) #u₁loopsym ∈ loopdependencies(op)
    # isunrolled₂ = isu₂unrolled(op)    
    falseexpr = Expr(:call, lv(:False)); trueexpr = Expr(:call, lv(:True)); rs = staticexpr(reg_size(ls));
    opp = first(parents(op))
    if (opp.instruction.instr === reductfunc) && isone(length(parents(opp)))
        opp = only(parents(opp))
    end
    isu₁, isu₂ = isunrolled_sym(opp, u₁loopsym, u₂loopsym, u₂max)
    u = isu₁ ? u₁ : 1
    mvar = Symbol(variable_name(opp, ifelse(isu₂, suffix, -1)), '_', u)
    if all(op.ref.loopedindex)
        inds = unrolledindex(op, ua, mask, inds_calc_by_ptr_offset)

        storeexpr = if reductfunc === Symbol("")
            Expr(:call, lv(:vstore!), vptr(op), mvar, inds)
        else
            Expr(:call, lv(:vstore!), lv(reductfunc), vptr(op), mvar, inds)
        end
        add_memory_mask!(storeexpr, op, ua, mask)
        push!(storeexpr.args, falseexpr, trueexpr, falseexpr, rs)
        push!(q.args, storeexpr)
    elseif u₁ > 1
        mvard = Symbol(mvar, "##data##")
        push!(q.args, Expr(:(=), mvard, Expr(:call, lv(:data), mvar)))
        for u ∈ 1:u₁
            mvaru = :(getfield($mvard, $u, false))
            inds = mem_offset_u(op, ua, inds_calc_by_ptr_offset, true, u-1)
            storeexpr = if reductfunc === Symbol("")
                Expr(:call, lv(:vstore!), vptr(op), mvaru, inds)
            else
                Expr(:call, lv(:vstore!), lv(reductfunc), vptr(op), mvaru, inds)
            end
            add_memory_mask!(storeexpr, op, ua, mask & ((u == u₁) | isvectorized(op)))
            push!(storeexpr.args, falseexpr, trueexpr, falseexpr, rs)
            push!(q.args, storeexpr)
        end
    else
        inds = mem_offset_u(op, ua, inds_calc_by_ptr_offset, true, 0)
        storeexpr = if reductfunc === Symbol("")
            Expr(:call, lv(:vstore!), vptr(op), mvar, inds)
        else
            Expr(:call, lv(:vstore!), lv(reductfunc), vptr(op), mvar, inds)
        end
        add_memory_mask!(storeexpr, op, ua, mask)
        push!(storeexpr.args, falseexpr, trueexpr, falseexpr, rs)
        push!(q.args, storeexpr)
    end
    nothing
end

function lower_tiled_store!(
    blockq::Expr, opsv1::Vector{Operation}, opsv2::Vector{Operation}, ls::LoopSet, unrollsyms::UnrollSymbols, u₁::Int, u₂::Int, mask::Bool
)
    ua = UnrollArgs(u₁, unrollsyms, u₂, u₂)
    for opsv ∈ (opsv1, opsv2)
        for op ∈ opsv
            lower_tiled_store!(blockq, op, ls, unrollsyms, u₁, u₂, mask)
        end
    end
end
# VectorizationBase implements optimizations for certain grouped stores
# thus we group stores together here to allow for these possibilities.
# (In particular, it tries to replace scatters with shuffles when there are groups
#   of stores offset from one another.)
function lower_tiled_store!(blockq::Expr, op::Operation, ls::LoopSet, unrollsyms::UnrollSymbols, u₁::Int, u₂::Int, mask::Bool)
    @unpack u₁loopsym, u₂loopsym, vectorized = unrollsyms
    reductfunc = storeinstr_preprend(op, vectorized)
    inds_calc_by_ptr_offset = indices_calculated_by_pointer_offsets(ls, op.ref)

    if (!((reductfunc === Symbol("")) && all(op.ref.loopedindex))) || (u₂ ≤ 1) || isconditionalmemop(op)
        # If we have a reductfunc, we're using a reducing store instead of a contiuguous or shuffle store anyway
        # so no benefit to being able to handle that case here, vs just calling the default `lower_store!` method
        for t ∈ 0:u₂-1
            unrollargs = UnrollArgs(u₁, unrollsyms, u₂, t)
            lower_store!(blockq, ls, op, unrollargs, mask, reductfunc, inds_calc_by_ptr_offset)
        end
        return
    end
    opp = first(parents(op))
    if (opp.instruction.instr === reductfunc) && isone(length(parents(opp)))
        throw("Operation $opp's instruction is $reductfunc, shouldn't be able to reach here.")
        # opp = only(parents(opp))
    end
    isu₁, isu₂ = isunrolled_sym(opp, u₁loopsym, u₂loopsym, u₂)
    @assert isu₂
    # It's reasonable forthis to be `!isu₁`
    u = Core.ifelse(isu₁, u₁, 1)
    tup = Expr(:tuple)
    for t ∈ 0:u₂-1
        mvar = Symbol(variable_name(opp, t), '_', u)
        push!(tup.args, mvar)
    end
    vut = :(VecUnroll($tup)) # `VecUnroll` of `VecUnroll`s
    ua = UnrollArgs(u₁, unrollsyms, u₂, 0)
    inds = mem_offset_u(op, ua, inds_calc_by_ptr_offset, false)
    unrollcurl₂ = unrolled_curly(op, u₂, u₂loopsym, vectorized, mask)
    falseexpr = Expr(:call, lv(:False)); trueexpr = Expr(:call, lv(:True)); rs = staticexpr(reg_size(ls));
    if isu₁ && u₁ > 1 # both unrolled
        unrollcurl₁ = unrolled_curly(op, u₁, u₁loopsym, vectorized, mask)
        inds = Expr(:call, unrollcurl₁, inds)
    end
    uinds = Expr(:call, unrollcurl₂, inds)
    storeexpr = Expr(:call, lv(:vstore!), vptr(op), vut, uinds)
    if mask && isvectorized(op)
        # add_memory_mask!(storeexpr, op, ua, mask)
        # we checked for `isconditionalmemop` earlier, so we skip this check
        # and just directly take the branch in `add_memory_mask!`
        push!(storeexpr.args, mask)
    end
    push!(storeexpr.args, falseexpr, trueexpr, falseexpr, rs)
    push!(blockq.args, storeexpr)
    nothing
end
