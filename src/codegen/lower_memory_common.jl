function parentind(ind::Symbol, op::Operation)
    for (id,opp) ∈ enumerate(parents(op))
        name(opp) === ind && return id
    end
    -1
end
function symbolind(ind::Symbol, op::Operation, td::UnrollArgs)
    id = parentind(ind, op)
    id == -1 && return ind, op
    @unpack u₁, u₁loopsym, u₂loopsym, u₂max, suffix = td
    parent = parents(op)[id]
    pvar, u₁op, u₂op = variable_name_and_unrolled(parent, u₁loopsym, u₂loopsym, u₂max, suffix)
    # pvar = if u₂loopsym ∈ loopdependencies(parent)
    #     variable_name(parent, suffix)
    # else
    #     mangledvar(parent)
    # end
    u = u₁op ? u₁ : 1
    ex = Symbol(pvar, '_', u)
    Expr(:call, lv(:staticm1), ex), parent
end

staticexpr(x::Int) = Expr(:call, Expr(:curly, lv(:Static), x))
staticexpr(x::Union{Symbol,Expr}) = Expr(:call, lv(:Static), x)
lazymulexpr(x::Int, y::Int) = staticexpr(x*y)
lazymulexpr(x::Int, y) = Expr(:call, lv(:lazymul), staticexpr(x), y)
maybestatic(x::Int) = staticexpr(x)
maybestatic(x::Union{Symbol,Expr}) = x

_MMind(ind::Union{Symbol,Expr}, str::Int) = Expr(:call, lv(:MM), VECTORWIDTHSYMBOL, ind, staticexpr(str))
_MMind(ind::Int, str::Int) = Expr(:call, lv(:MM), VECTORWIDTHSYMBOL, staticexpr(ind), staticexpr(str))

_MMind(ind::Union{Int,Symbol,Expr}, str::Union{Symbol,Expr}) = addexpr(mulexpr(_MMind(0,1),str),ind)
_MMind(ind, str::MaybeKnown) = isknown(str) ? _MMind(ind, gethint(str)) : _MMind(ind, getsym(str))

function addoffset!(ret::Expr, ex, stride, offset::Int, _mm::Bool)
    # This code could be orthogonal, but because `ex` might not be an `Expr` (e.g., could be a Symbol)
    # I want to avoid type instabilities. Hence, the branchiness of the code here.
    if iszero(offset)
            # `ind` wouldn't be an `Expr`, so we just return
        if _mm
            push!(ret.args, _MMind(ex, stride))
        else
            push!(ret.args, ex)
        end
        return
    end
    _ind::Expr = addexpr(ex, offset)
    push!(ret.args, _mm ? _MMind(_ind, stride) : _ind)
    nothing
end
function addoffset!(ret::Expr, stride, offset::Int, _mm::Bool)
    push!(ret.args, _mm ? _MMind(offset, stride) : staticexpr(offset))
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
    strides = getstrides(op)
    loopedindex = op.ref.loopedindex
    # inds_calc_by_ptr_offset = indices_calculated_by_pointer_offsets(ls, op.ref)
    @unpack vloopsym = td
    for (n,ind) ∈ enumerate(indices)
        indvectorized = _mm & (ind === vloopsym)
        offset = offsets[n] % Int
        stride = strides[n] % Int
        if indvectorized
            @unpack vstep = td
            if isknown(vstep)
                stride *= gethint(vstep)
            else
                if isone(stride)
                    if inds_calc_by_ptr_offset[n]
                        addoffset!(ret, getsym(vstep), offset, true)
                    else
                        addoffset!(ret, ind, getsym(vstep), offset, true)
                    end
                else
                    if inds_calc_by_ptr_offset[n]
                        addoffset!(ret, lazymulexpr(getsym(vstep), stride), offset, true)
                    else
                        addoffset!(ret, ind, lazymulexpr(getsym(vstep), stride), offset, true)
                    end
                end
                continue
            end
        end
        # if ind isa Int # impossible
            # push!(ret.args, ind + offset)
        # else
        if loopedindex[n]
            if inds_calc_by_ptr_offset[n]
                addoffset!(ret, stride, offset, indvectorized)
            else
                addoffset!(ret, ind, stride, offset, indvectorized)
            end
        else
            newname, parent = symbolind(ind, op, td)
            # _mmi = indvectorized && parent !== op && (!isvectorized(parent))
            # addoffset!(ret, newname, stride, offset, _mmi)
            _mmi = indvectorized && parent !== op && (!isvectorized(parent))
            @assert !_mmi "Please file an issue with an example of how you got this."
            addoffset!(ret, newname, 1, offset, false)
        end
    end
    ret
end
isconditionalmemop(op::Operation) = (instruction(op).instr === :conditionalload) || (instruction(op).instr === :conditionalstore!)
# function unrolled_curly(op::Operation, u₁::Int, u₁loopsym::Symbol, vectorized::Symbol, mask::Bool)
function unrolled_curly(op::Operation, u₁::Int, u₁loop::Loop, vloop::Loop, mask::Bool)
    u₁loopsym = u₁loop.itersymbol
    vloopsym = vloop.itersymbol
    indices = getindicesonly(op)
    vstep = step(vloop)
    # loopedindex = op.ref.loopedindex
    # @assert all(loopedindex)
    # @unpack u₁, u₁loopsym, vloopsym = td
    # @show vptr(op), inds_calc_by_ptr_offset
    # isone(u₁) && return mem_offset_u(op, td, inds_calc_by_ptr_offset, true)
    AV = AU = -1
    for (n,ind) ∈ enumerate(indices)
        if ind === vloopsym
            AV = n
        end
        if ind === u₁loopsym
            AU = n
        end
    end
    # if AU == -1
    #     return mem_offset_u(op, td, inds_calc_by_ptr_offset, true)
    # end
    # ind = mem_offset_u(op, td, inds_calc_by_ptr_offset, false)
    vecnotunrolled = AU != AV
    conditional_memory_op = isconditionalmemop(op)
    if mask || conditional_memory_op
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
    @assert isknown(step(u₁loop)) "Unrolled loops must have known steps to use `Unroll` type; this is a bug, shouldn't have reached here"
    if AV > 0
        @assert isknown(step(vloop)) "Vectorized loops must have known steps to use `Unroll` type; this is a bug, shouldn't have reached here."
        X = convert(Int, getstrides(op)[AV])
        X *= gethint(step(vloop))
        intvecsym = :(Int($VECTORWIDTHSYMBOL))
        if vecnotunrolled
            # Expr(:call, Expr(:curly, lv(:Unroll), AU, 1, u₁, AV, intvecsym, M, 1), ind)
            Expr(:curly, lv(:Unroll), AU, gethint(u₁loop), u₁, AV, intvecsym, M, X)
        else
            if isone(step(u₁loop))
                Expr(:curly, lv(:Unroll), AU, intvecsym, u₁, AV, intvecsym, M, X)
            else
                unrollstepexpr = :(Int($(mulexpr(VECTORWIDTHSYMBOL, step(u₁loop)))))
                Expr(:curly, lv(:Unroll), AU, unrollstepexpr, u₁, AV, intvecsym, M, X)
            end
        end
    else
        Expr(:curly, lv(:Unroll), AU, gethint(u₁loop), u₁, AV, 1, M, 1)
    end
end
function unrolledindex(op::Operation, td::UnrollArgs, mask::Bool, inds_calc_by_ptr_offset::Vector{Bool})
    @unpack u₁, u₁loopsym, u₁loop, vloop = td
    isone(u₁) && return mem_offset_u(op, td, inds_calc_by_ptr_offset, true)
    any(==(u₁loopsym), getindicesonly(op)) || return mem_offset_u(op, td, inds_calc_by_ptr_offset, true)
    
    unrollcurl = unrolled_curly(op, u₁, u₁loop, vloop, mask)
    ind = mem_offset_u(op, td, inds_calc_by_ptr_offset, false)
    Expr(:call, unrollcurl, ind)
end

function add_vectorized_offset!(ret::Expr, ind, stride, offset, incr, _mm::Bool)
    if isone(incr)
        if iszero(offset)
            _ind = Expr(:call, lv(:vadd_fast), VECTORWIDTHSYMBOL, maybestatic(ind))
        else
            _ind = Expr(:call, lv(:vadd_fast), ind, Expr(:call, lv(:vadd_fast), VECTORWIDTHSYMBOL, staticexpr(offset)))
        end
        if _mm
            _ind = _MMind(_ind, stride)
        end
        push!(ret.args, _ind)
    elseif iszero(incr)
        if iszero(offset)
            if _mm
                push!(ret.args, _MMind(ind, stride))
            else
                push!(ret.args, ind)
            end
        else
            addoffset!(ret, ind, stride, offset, _mm)
        end
    else
        if iszero(offset)
            _ind = Expr(:call, lv(:vadd_fast), Expr(:call, lv(:vmul_fast), VECTORWIDTHSYMBOL, maybestatic(incr)), maybestatic(ind))
        else
            _ind = Expr(:call, lv(:vadd_fast), ind, Expr(:call, lv(:vadd_fast), Expr(:call, lv(:vmul_fast), VECTORWIDTHSYMBOL, maybestatic(incr)), staticexpr(offset)))
        end
        if _mm
            _ind = _MMind(_ind, stride)
        end
        push!(ret.args, _ind)
    end
end


function add_vectorized_offset_unrolled!(ret::Expr, stride, offset, incr, _mm::Bool)
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
        _ind = _MMind(_ind, stride)
    end
    push!(ret.args, _ind)
    nothing
end
function add_vectorized_offset!(ret::Expr, ind, stride, offset, incr, unrolled, _mm::Bool)
    if unrolled
        add_vectorized_offset_unrolled!(ret, stride, offset, incr, _mm)
    else
        add_vectorized_offset!(ret, ind, stride, offset, incr, _mm)
    end
end

function _add_unrolled_offset!(ret::Expr, incr, ind, offset, ind_by_offset, ustep::MaybeKnown, _mm::Bool)
    if isknown(ustep)
        incr *= gethint(ustep)
        if indvectorized
            add_vectorized_offset!(ret, ind, offset, incr, ind_by_offset, _mm)
        elseif ind_by_offset
            addoffset!(ret, incr + offset, false)
        else
            addoffset!(ret, ind, incr + offset, false)
        end
    else
        lm_ind = lazymulexpr(incr, getsym(ustep))
        if indvectorized
            add_vectorized_offset!(ret, ind, offset, lm_ind, ind_by_offset, _mm)
        elseif ind_by_offset
            addoffset!(ret, lm_ind, offset, false)
        else
            addoffset!(ret, lm_ind, offset, false)
            addoffset!(ret, addexpr(lm_ind, ind), offset, false)
        end
    end
end

function mem_offset_u(op::Operation, td::UnrollArgs, inds_calc_by_ptr_offset::Vector{Bool}, _mm::Bool, incr₁::Int = 0)
    @assert accesses_memory(op) "Computing memory offset only makes sense for operations that access memory."
    @unpack u₁loopsym, u₂loopsym, vloopsym, u₁step, u₂step, vstep, suffix = td
    
    #u₁
    incr₂ = max(suffix, 0)#(suffix == -1) ? 0 : suffix
    ret = Expr(:tuple)
    indices = getindicesonly(op)
    offsets = getoffsets(op)
    strides = getstrides(op)
    # allbasezero = all(inds_calc_by_ptr_offset) && all(iszero, offsets)
    loopedindex = op.ref.loopedindex
    if iszero(incr₁) & iszero(incr₂)
        return mem_offset(op, td, inds_calc_by_ptr_offset, _mm)
        # append_inds!(ret, indices, loopedindex)
    else
        for (n,ind) ∈ enumerate(indices)
            ind_by_offset = inds_calc_by_ptr_offset[n]
            offset = convert(Int, offsets[n])
            stride = convert(Int, strides[n])
            indvectorized = ind === vloopsym
            if ind === u₁loopsym
                _add_unrolled_offset!(ret, incr₁*stride, ind, offset, ind_by_offset, u₁step, _mm)
            elseif ind === u₂loopsym
                _add_unrolled_offset!(ret, incr₂*stride, ind, offset, ind_by_offset, u₂step, _mm)
            elseif loopedindex[n]
                if _mm & indvectorized
                    if isknown(vstep)
                        stride *= gethint(vstep)
                        if ind_by_offset
                            addoffset!(ret, stride, offset, _mm & indvectorized)
                        else
                            addoffset!(ret, ind, stride, offset, _mm & indvectorized)
                        end
                    else
                        strideexpr = mulexpr(stride,getsym(vstep))
                        if ind_by_offset
                            addoffset!(ret, strideexpr, offset, _mm & indvectorized)
                        else
                            addoffset!(ret, ind, strideexpr, offset, _mm & indvectorized)
                        end
                    end
                else
                    # stride doesn't matter
                    if ind_by_offset
                        addoffset!(ret, 1, offset, _mm & indvectorized)
                    else
                        addoffset!(ret, ind, 1, offset, _mm & indvectorized)
                    end
                end
            else
                newname, parent = symbolind(ind, op, td)
                # _mmi = _mm && indvectorized && parent !== op && (!isvectorized(parent))
                #                              addoffset!(ret, newname, 1, offset, _mmi)
                @assert !_mmi "Please file an issue with an example of how you got this."
                addoffset!(ret, newname, 1, offset, false)
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

function add_memory_mask!(memopexpr::Expr, op::Operation, td::UnrollArgs, mask::Bool)
    @unpack u₁, u₁loopsym, u₂loopsym, vloopsym, u₂max, suffix = td
    if isconditionalmemop(op)
        condop = last(parents(op))
        opu₂ = (suffix ≠ -1) && isu₂unrolled(op)
        condvar, condu₁unrolled = condvarname_and_unroll(condop, u₁loopsym, u₂loopsym, u₂max, suffix, opu₂)
        # if it isn't unrolled, then `m`
        u = condu₁unrolled ? u₁ : 1
        # u = isu₁unrolled(condop) ? u₁ : 1
        condvar = Symbol(condvar, '_', u)
        # @show condvar
        if !mask || (!isvectorized(op))
            push!(memopexpr.args, condvar)
        else
            # we only want to apply mask to `u₁`
            push!(memopexpr.args, Expr(:call, lv(:and_last), condvar, MASKSYMBOL))
        end
    elseif mask && isvectorized(op)
        push!(memopexpr.args, MASKSYMBOL)
    end
end

varassignname(var::Symbol, u::Int, isunrolled::Bool) = isunrolled ? Symbol(var, u) : var
# name_memoffset only gets called when vectorized
function name_memoffset(var::Symbol, op::Operation, td::UnrollArgs, u₁unrolled::Bool, inds_calc_by_ptr_offset::Vector{Bool})
    @unpack u₁, u₁loopsym, u₂loopsym, suffix = td
    if (suffix == -1) && u₁ < 0 # u₁ == -1 sentinel value meaning not unrolled
        name = var
        mo = mem_offset(op, td, inds_calc_by_ptr_offset, true)
    else
        name = u₁unrolled ? Symbol(var, u₁) : var
        mo = mem_offset_u(op, td, inds_calc_by_ptr_offset)
    end
    name, mo
end

function condvarname_and_unroll(cond, u₁loop, u₂loop, u₂max, suffix, opu₂)
    condvar, condu₁, condu₂ = variable_name_and_unrolled(cond, u₁loop, u₂loop, u₂max, Core.ifelse(opu₂, suffix, -1))
    condvar, condu₁
end
