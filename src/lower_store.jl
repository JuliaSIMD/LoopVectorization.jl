
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

# pvariable_name(op::Operation, ::Nothing) = mangledvar(first(parents(op)))
# pvariable_name(op::Operation, ::Nothing, ::Symbol) = mangledvar(first(parents(op)))
# pvariable_name(op::Operation, suffix) = Symbol(pvariable_name(op, nothing), suffix, :_)
# function pvariable_name(op::Operation, suffix, tiled::Symbol)
#     parent = first(parents(op))
#     mname = mangledvar(parent)
#     tiled ∈ loopdependencies(parent) ? Symbol(mname, suffix, :_) : mname
# end


# function lower_conditionalstore_scalar!(
#     q::Expr, op::Operation, ua::UnrollArgs, mask::Union{Nothing,Symbol,Unsigned}, inds_calc_by_ptr_offset::Vector{Bool}, rsi::Int
# )
#     @unpack u₁, u₁loopsym, u₂loopsym, vectorized, suffix = ua
#     mvar, opu₁, opu₂ = variable_name_and_unrolled(first(parents(op)), u₁loopsym, u₂loopsym, suffix)
#     # var = pvariable_name(op, suffix, tiled)
#     cond = last(parents(op))
#     condvar, condu₁ = condvarname_and_unroll(cond, u₁loopsym, u₂loopsym, suffix, opu₂)
#     loopdeps = loopdependencies(op)
#     opu₁ = u₁loopsym ∈ loopdeps
#     ptr = vptr(op)
#     storef = storeinstr(op, vectorized)
#     falseexpr = Expr(:call, lv(:False)); trueexpr = Expr(:call, lv(:True)); rs = staticexpr(rsi);
#     for u ∈ 0:u₁-1
#         varname = varassignname(mvar, u, opu₁)
#         condvarname = varassignname(condvar, u, condu₁)
#         td = UnrollArgs(ua, u)
#         storecall = Expr(:call, storef, ptr, varname, mem_offset_u(op, td, inds_calc_by_ptr_offset), falseexpr, trueexpr, falseexpr, rs)
#         push!(q.args, Expr(:&&, condvarname, storecall))
#     end
#     nothing
# end
# function lower_conditionalstore_vectorized!(
#     q::Expr, op::Operation, ua::UnrollArgs, mask::Union{Nothing,Symbol,Unsigned}, isunrolled::Bool, inds_calc_by_ptr_offset::Vector{Bool}, rsi::Int
# )
#     @unpack u₁, u₁loopsym, u₂loopsym, vectorized, suffix = ua
#     loopdeps = loopdependencies(op)
#     @assert vectorized ∈ loopdeps
#     mvar, opu₁, opu₂ = variable_name_and_unrolled(first(parents(op)), u₁loopsym, u₂loopsym, suffix)
#     # var = pvariable_name(op, suffix, tiled)
#     if isunrolled
#         umin = 0
#         U = u₁
#     else
#         umin = -1
#         U = 0
#     end
#     ptr = vptr(op)
#     vecnotunrolled = vectorized !== u₁loopsym
#     cond = last(parents(op))
#     condvar, condu₁ = condvarname_and_unroll(cond, u₁loopsym, u₂loopsym, suffix, opu₂)
#     # @show parents(op) cond condvar
#     storef = storeinstr(op, vectorized)
#     falseexpr = Expr(:call, lv(:False)); trueexpr = Expr(:call, lv(:True)); rs = staticexpr(rsi);
#     for u ∈ 0:U-1
#         td = UnrollArgs(ua, u)
#         name, mo = name_memoffset(mvar, op, td, opu₁, inds_calc_by_ptr_offset)
#         condvarname = varassignname(condvar, u, condu₁)
#         instrcall = Expr(:call, storef, ptr, name, mo)
#         if mask !== nothing && (vecnotunrolled || u == U - 1)
#             push!(instrcall.args, Expr(:call, :&, condvarname, mask))
#         else
#             push!(instrcall.args, condvarname)
#         end
#         push!(instrcall.args, falseexpr, trueexpr, falseexpr, rs)
#         push!(q.args, instrcall)
#     end
# end

# function lower_store_scalar!(
#     q::Expr, op::Operation, ua::UnrollArgs, mask::Union{Nothing,Symbol,Unsigned}, inds_calc_by_ptr_offset::Vector{Bool}, rsi::Int
# )
#     @unpack u₁, u₁loopsym, u₂loopsym, vectorized, suffix = ua
#     mvar, opu₁, opu₂ = variable_name_and_unrolled(first(parents(op)), u₁loopsym, u₂loopsym, suffix)
#     ptr = vptr(op)
#     # var = pvariable_name(op, suffix, tiled)
#     # parentisunrolled = unrolled ∈ loopdependencies(first(parents(op)))
#     storef = storeinstr(op, vectorized)
#     falseexpr = Expr(:call, lv(:False)); trueexpr = Expr(:call, lv(:True)); rs = staticexpr(rsi);
#     for u ∈ 0:u₁-1
#         varname = varassignname(mvar, u, opu₁)
#         td = UnrollArgs(ua, u)
#         storecall = Expr(:call, storef, ptr, varname, mem_offset_u(op, td, inds_calc_by_ptr_offset), falseexpr, trueexpr, falseexpr, rs)
#         push!(q.args, storecall)
#     end
#     nothing
# end
# function lower_store_vectorized!(
#     q::Expr, op::Operation, ua::UnrollArgs, mask::Union{Nothing,Symbol,Unsigned}, isunrolled::Bool, inds_calc_by_ptr_offset::Vector{Bool}, rsi::Int
# )
#     @unpack u₁, u₁loopsym, u₂loopsym, vectorized, suffix = ua
#     loopdeps = loopdependencies(op)
#     @assert vectorized ∈ loopdeps
#     mvar, opu₁, opu₂ = variable_name_and_unrolled(first(parents(op)), u₁loopsym, u₂loopsym, suffix)
#     ptr = vptr(op)
#     # var = pvariable_name(op, suffix, tiled)
#     # parentisunrolled = unrolled ∈ loopdependencies(first(parents(op)))
#     if isunrolled
#         umin = 0
#         U = u₁
#     else
#         umin = -1
#         U = 0
#     end
#     vecnotunrolled = vectorized !== u₁loopsym
#     storef = storeinstr(op, vectorized)
#     falseexpr = Expr(:call, lv(:False)); trueexpr = Expr(:call, lv(:True)); rs = staticexpr(rsi);
#     for u ∈ umin:U-1
#         td = UnrollArgs(ua, u)
#         name, mo = name_memoffset(mvar, op, td, opu₁, inds_calc_by_ptr_offset)
#         instrcall = Expr(:call, storef, ptr, name, mo)
#         if mask !== nothing && (vecnotunrolled || u == U - 1)
#             push!(instrcall.args, mask)
#         end
#         push!(instrcall.args, falseexpr, trueexpr, falseexpr, rs)
#         push!(q.args, instrcall)
#     end
# end
# function _lower_store_nonloopindex!()
    
# end

function lower_store!(
    q::Expr, ls::LoopSet, op::Operation, ua::UnrollArgs, mask::Union{Nothing,Symbol,Unsigned} = nothing
)
    @unpack u₁, u₁loopsym, u₂loopsym, vectorized, u₂max, suffix = ua
    isunrolled₁ = isu₁unrolled(op) #u₁loopsym ∈ loopdependencies(op)
    # isunrolled₂ = isu₂unrolled(op)
    inds_calc_by_ptr_offset = indices_calculated_by_pointer_offsets(ls, op.ref)
    falseexpr = Expr(:call, lv(:False)); trueexpr = Expr(:call, lv(:True)); rs = staticexpr(reg_size(ls));

    reductfunc = storeinstr_preprend(op, vectorized)
    opp = first(parents(op))
    if (opp.instruction.instr === reductfunc) && isone(length(parents(opp)))
        opp = only(parents(opp))
    end
    isu₁, isu₂ = isunrolled_sym(opp, u₁loopsym, u₂loopsym, u₂max)
    u = isu₁ ? u₁ : 1
    mvar = if isu₂
        Symbol(variable_name(opp, suffix), '_', u)
    else
        Symbol(variable_name(opp, nothing), '_', u)
    end

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
            if (mask === nothing) || (u == u₁) || isvectorized(op)
                add_memory_mask!(storeexpr, op, ua, mask)
            else
                add_memory_mask!(storeexpr, op, ua, nothing)
            end
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


