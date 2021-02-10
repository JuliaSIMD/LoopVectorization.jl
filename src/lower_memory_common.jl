function parentind(ind::Symbol, op::Operation)
    for (id,opp) ∈ enumerate(parents(op))
        name(opp) === ind && return id
    end
    -1
end
function symbolind(ind::Symbol, op::Operation, td::UnrollArgs)
    id = parentind(ind, op)
    id == -1 && return ind, op
    @unpack u₁, u₁loopsym, u₂loopsym, suffix = td
    parent = parents(op)[id]
    pvar = if u₂loopsym ∈ loopdependencies(parent)
        variable_name(parent, suffix)
    else
        mangledvar(parent)
    end
    ex = u₁loopsym ∈ loopdependencies(parent) ? Symbol(pvar, '_', u₁) : pvar
    Expr(:call, lv(:staticm1), ex), parent
end

staticexpr(x::Integer) = Expr(:call, Expr(:curly, lv(:Static), convert(Int, x)))
staticexpr(x) = Expr(:call, lv(:Static), x)
maybestatic(x::Integer) = staticexpr(x)
maybestatic(x) = x
_MMind(ind) = Expr(:call, lv(:MM), VECTORWIDTHSYMBOL, ind)
_MMind(ind::Integer) = Expr(:call, lv(:MM), VECTORWIDTHSYMBOL, staticexpr(ind))
function addoffset!(ret::Expr, ex, offset::Integer, _mm::Bool)
    if iszero(offset)
        if _mm
            ind = _MMind(ex)
        else
            push!(ret.args, ex)
            return
        end
    elseif _mm
        ind = _MMind(Expr(:call, lv(:vadd_fast), ex, staticexpr(offset)))
    else
        ind = Expr(:call, lv(:vadd_fast), ex, staticexpr(offset))
    end
    push!(ret.args, ind)
    nothing
end
function addoffset!(ret::Expr, offset::Int, _mm::Bool)
    push!(ret.args, _mm ? _MMind(offset) : staticexpr(offset))
    nothing
end

"""
unrolled loads are calculated as offsets with respect to an initial gesp. This has proven important to helping LLVM generate efficient code in some cases.
Therefore, unrolled === true results in inds being ignored.
_mm means to insert `mm`s.
"""
function mem_offset(op::Operation, td::UnrollArgs, inds_calc_by_ptr_offset::Vector{Bool}, _mm::Bool)
    # @assert accesses_memory(op) "Computing memory offset only makes sense for operations that access memory."
    ret = Expr(:tuple)
    indices = getindicesonly(op)
    offsets = getoffsets(op)
    loopedindex = op.ref.loopedindex
    # inds_calc_by_ptr_offset = indices_calculated_by_pointer_offsets(ls, op.ref)
    @unpack vectorized = td
    for (n,ind) ∈ enumerate(indices)
        indvectorized = _mm & (ind === vectorized)
        offset = offsets[n] % Int
        # if ind isa Int # impossible
            # push!(ret.args, ind + offset)
        # else
        if loopedindex[n]
            if inds_calc_by_ptr_offset[n]
                addoffset!(ret, offset, indvectorized)
            else
                addoffset!(ret, ind, offset, indvectorized)
            end
        else
            newname, parent = symbolind(ind, op, td)
            _mmi = indvectorized && parent !== op && (!isvectorized(parent))
            addoffset!(ret, newname, offset, _mmi)
        end
    end
    ret
end
isconditionalmemop(op::Operation) = (instruction(op).instr === :conditionalload) || (instruction(op).instr === :conditionalstore!)
function unrolledindex(op::Operation, td::UnrollArgs, mask, inds_calc_by_ptr_offset::Vector{Bool})
    indices = getindicesonly(op)
    loopedindex = op.ref.loopedindex
    # @assert all(loopedindex)
    @unpack u₁, u₁loopsym, u₂loopsym, vectorized, suffix = td
    # @show vptr(op), inds_calc_by_ptr_offset
    isone(u₁) && return mem_offset_u(op, td, inds_calc_by_ptr_offset, true)
    AV = AU = -1
    for (n,ind) ∈ enumerate(indices)
        if ind === vectorized
            AV = n
        end
        if ind === u₁loopsym
            AU = n
        end
    end
    if AU == -1
        return mem_offset_u(op, td, inds_calc_by_ptr_offset, true)
    end
    ind = mem_offset_u(op, td, inds_calc_by_ptr_offset, false)
    vecnotunrolled = AU != AV
    conditional_memory_op = isconditionalmemop(op)
    if conditional_memory_op || (mask !== nothing)
        M = one(UInt)
        # `isu₁unrolled(last(parents(op)))` === is condop unrolled?
        # isu₁unrolled(last(parents(op)))
        if vecnotunrolled || conditional_memory_op # mask all
            M = (M << u₁) - M
        else # mask last
            M <<= (u₁ - 1)
        end
    else
        M = zero(UInt)
    end
    if AV > 0
        intvecsym = :(Int($VECTORWIDTHSYMBOL))
        if vecnotunrolled
            Expr(:call, Expr(:curly, lv(:Unroll), AU, 1, u₁, AV, intvecsym, M, 1), ind)
        else
            Expr(:call, Expr(:curly, lv(:Unroll), AU, intvecsym, u₁, AV, intvecsym, M, 1), ind)
        end
    else
        Expr(:call, Expr(:curly, lv(:Unroll), AU, 1, u₁, AV, 1, M, 1), ind)
    end
end

function add_vectorized_offset!(ret::Expr, ind, offset, incr, _mm::Bool)
    if isone(incr)
        if iszero(offset)
            _ind = Expr(:call, lv(:vadd_fast), VECTORWIDTHSYMBOL, maybestatic(ind))
        else
            _ind = Expr(:call, lv(:vadd_fast), ind, Expr(:call, lv(:vadd_fast), VECTORWIDTHSYMBOL, staticexpr(offset)))
        end
        if _mm
            _ind = _MMind(_ind)
        end
        push!(ret.args, _ind)
    elseif iszero(incr)
        if iszero(offset)
            if _mm
                push!(ret.args, _MMind(ind))
            else
                push!(ret.args, ind)
            end
        else
            addoffset!(ret, ind, offset, _mm)
        end
    else
        if iszero(offset)
            _ind = Expr(:call, lv(:vadd_fast), Expr(:call, lv(:vmul_fast), VECTORWIDTHSYMBOL, maybestatic(incr)), maybestatic(ind))
        else
            _ind = Expr(:call, lv(:vadd_fast), ind, Expr(:call, lv(:vadd_fast), Expr(:call, lv(:vmul_fast), VECTORWIDTHSYMBOL, maybestatic(incr)), staticexpr(offset)))
        end
        if _mm
            _ind = _MMind(_ind)
        end
        push!(ret.args, _ind)
    end
end
function add_vectorized_offset_unrolled!(ret::Expr, offset, incr, _mm::Bool)
    _ind = if isone(incr)
        if iszero(offset)
            Expr(:call, lv(:Static), VECTORWIDTHSYMBOL)
        else
            Expr(:call, lv(:vadd_fast), VECTORWIDTHSYMBOL, staticexpr(offset))
        end
    elseif iszero(incr)
        if iszero(offset)
            Expr(:call, lv(:Zero))
        else
            staticexpr(offset)
        end
    elseif iszero(offset)
        Expr(:call, lv(:*), VECTORWIDTHSYMBOL, maybestatic(incr))
    else
        Expr(:call, lv(:vadd_fast), Expr(:call, lv(:vmul_fast), VECTORWIDTHSYMBOL, maybestatic(incr)), staticexpr(offset))
    end
    if _mm
        _ind = _MMind(_ind)
    end
    push!(ret.args, _ind)
    nothing
end
function add_vectorized_offset!(ret::Expr, ind, offset, incr, unrolled, _mm::Bool)
    if unrolled
        add_vectorized_offset_unrolled!(ret, offset, incr, _mm)
    else
        add_vectorized_offset!(ret, ind, offset, incr, _mm)
    end
end
function mem_offset_u(op::Operation, td::UnrollArgs, inds_calc_by_ptr_offset::Vector{Bool}, _mm::Bool, incr₁::Int = 0)
    @assert accesses_memory(op) "Computing memory offset only makes sense for operations that access memory."
    @unpack u₁loopsym, u₂loopsym, vectorized, suffix = td
    #u₁
    incr₂ = isnothing(suffix) ? 0 : suffix
    ret = Expr(:tuple)
    indices = getindicesonly(op)
    offsets = getoffsets(op)
    # allbasezero = all(inds_calc_by_ptr_offset) && all(iszero, offsets)
    loopedindex = op.ref.loopedindex
    if iszero(incr₁) & iszero(incr₂)
        return mem_offset(op, td, inds_calc_by_ptr_offset, _mm)
        # append_inds!(ret, indices, loopedindex)
    else
        for (n,ind) ∈ enumerate(indices)
            ind_by_offset = inds_calc_by_ptr_offset[n]
            offset = convert(Int, offsets[n])
            indvectorized = ind === vectorized
            # if n == zero_offset
                # addoffset!(ret, 0)
            # elseif ind === u₁loopsym
            # if vptr(op) === Symbol("##vptr##_A")
            #     # @show ind, u₁loopsym, u₂loopsym, ind_by_offset, indvectorized
            #     @show ind, offset, incr₁, ind_by_offset, _mm
            # end
            if ind === u₁loopsym
                if indvectorized
                    add_vectorized_offset!(ret, ind, offset, incr₁, ind_by_offset, _mm)
                elseif ind_by_offset
                    addoffset!(ret, incr₁ + offset, false)
                else
                    addoffset!(ret, ind, incr₁ + offset, false)
                end
            elseif ind === u₂loopsym
                if indvectorized
                    add_vectorized_offset!(ret, ind, offset, incr₂, ind_by_offset, _mm)
                elseif ind_by_offset
                    addoffset!(ret, incr₂ + offset, false)
                else
                    addoffset!(ret, ind, incr₂ + offset, false)
                end
            elseif loopedindex[n]
                if ind_by_offset
                    addoffset!(ret, offset, _mm & indvectorized)
                else
                    addoffset!(ret, ind, offset, _mm & indvectorized)
                end
            else
                newname, parent = symbolind(ind, op, td)
                _mmi = _mm && indvectorized && parent !== op && (!isvectorized(parent))
                addoffset!(ret, newname, offset, _mmi)
            end
        end
    end
    ret
end

@inline and_last(a, b) = a & b
@generated function and_last(v::VecUnroll{N}, m) where {N}
    q = Expr(:block, Expr(:meta,:inline), :(vd = data(v)))
    t = Expr(:tuple)
    for n ∈ 1:N
        push!(t.args, :(getfield(vd, $n, false)))
    end
    push!(t.args, :(getfield(vd, $(N+1), false) & m))
    push!(q.args, Expr(:call, lv(:VecUnroll), t))
    q
end

function add_memory_mask!(memopexpr::Expr, op::Operation, td::UnrollArgs, mask::Union{Nothing,Symbol,Unsigned})
    @unpack u₁, u₁loopsym, u₂loopsym, vectorized, suffix = td
    if isconditionalmemop(op)
        condop = last(parents(op))
        opu₂ = !isnothing(suffix) && isu₂unrolled(op)
        condvar = condvarname_and_unroll(condop, u₁loopsym, u₂loopsym, suffix, opu₂)
        # if it isn't unrolled, then `m`
        u = isu₁unrolled(condop) ? u₁ : 1
        condvar = Symbol(condvar, '_', u)
        # @show condvar
        if mask === nothing || (!isvectorized(op))
            push!(memopexpr.args, condvar)
        else
            # we only want to apply mask to `u₁`
            push!(memopexpr.args, Expr(:call, lv(:and_last), condvar, mask))
        end
    elseif mask !== nothing && isvectorized(op)
        push!(memopexpr.args, mask)
    end
end

function varassignname(var::Symbol, u::Int, isunrolled::Bool)
    isunrolled ? Symbol(var, u) : var
end
# name_memoffset only gets called when vectorized
function name_memoffset(var::Symbol, op::Operation, td::UnrollArgs, u₁unrolled::Bool, inds_calc_by_ptr_offset::Vector{Bool})
    @unpack u₁, u₁loopsym, u₂loopsym, suffix = td
    if isnothing(suffix) && u₁ < 0 # u₁ == -1 sentinel value meaning not unrolled
        name = var
        mo = mem_offset(op, td, inds_calc_by_ptr_offset, true)
    else
        name = u₁unrolled ? Symbol(var, u₁) : var
        mo = mem_offset_u(op, td, inds_calc_by_ptr_offset)
    end
    name, mo
end

function condvarname_and_unroll(cond, u₁loop, u₂loop, suffix, opu₂)
    if isnothing(suffix) || !opu₂
        condvar, condu₁, condu₂ = variable_name_and_unrolled(cond, u₁loop, u₂loop, nothing)
    else
        condvar, condu₁, condu₂ = variable_name_and_unrolled(cond, u₁loop, u₂loop, suffix)
    end
    condvar#, condu₁
end
