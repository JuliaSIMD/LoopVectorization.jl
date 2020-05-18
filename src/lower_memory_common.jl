
function parentind(ind::Symbol, op::Operation)
    for (id,opp) ∈ enumerate(parents(op))
        name(opp) === ind && return id
    end
    -1
end
function symbolind(ind::Symbol, op::Operation, td::UnrollArgs)
    id = parentind(ind, op)
    id == -1 && return ind
    @unpack u₁, u₁loopsym, u₂loopsym, suffix = td
    parent = parents(op)[id]
    pvar = if u₂loopsym ∈ loopdependencies(parent)
        variable_name(parent, suffix)
    else
        mangledvar(parent)
    end
    ex = u₁loopsym ∈ loopdependencies(parent) ? Symbol(pvar, u₁) : pvar
    Expr(:call, lv(:staticm1), ex)
end


_MMind(ind) = Expr(:call, lv(:_MM), VECTORWIDTHSYMBOL, ind)
_MMind(ind::Integer) = Expr(:call, lv(:_MM), VECTORWIDTHSYMBOL, convert(Int, ind))
function addoffset!(ret::Expr, ex, offset::Integer, _mm::Bool = false)
    if iszero(offset)
        if _mm
            ind = _MMind(ex)
        else
            push!(ret.args, ex)
            return
        end
    elseif _mm
        ind = _MMind(Expr(:call, lv(:vadd), ex, convert(Int, offset)))
    else
        ind = Expr(:call, lv(:vadd), ex, convert(Int, offset))
    end
    push!(ret.args, ind)
    nothing
end
function addoffset!(ret::Expr, offset::Integer, _mm::Bool = false)
    if iszero(offset)
        ex = Expr(:call, lv(:Zero))
        if _mm
            push!(ret.args, _MMind(ex))
        else
            push!(ret.args, ex)
        end
    elseif _mm
        push!(ret.args, _MMind(convert(Int, offset)))
    else
        push!(ret.args, convert(Int, offset))
    end
    nothing
end

"""
unrolled loads are calculated as offsets with respect to an initial gesp. This has proven important to helping LLVM generate efficient code in some cases.
Therefore, unrolled === true results in inds being ignored.
_mm means to insert `mm`s.
"""
function mem_offset(op::Operation, td::UnrollArgs, unrolled::Bool, _mm::Bool = true)
    # @assert accesses_memory(op) "Computing memory offset only makes sense for operations that access memory."
    ret = Expr(:tuple)
    indices = getindicesonly(op)
    offsets = getoffsets(op)
    loopedindex = op.ref.loopedindex
    unrolled = unrolled && use_gesp(op, td)
    @unpack vectorized = td
    for (n,ind) ∈ enumerate(indices)
        offset = _mm ? offsets[n] % Int : 0
        # if ind isa Int # impossible
            # push!(ret.args, ind + offset)
        # else
        if loopedindex[n]
            if unrolled
                addoffset!(ret, offset, _mm && ind === vectorized)
            else
                addoffset!(ret, ind, offset, _mm && ind === vectorized)
            end
        else
            addoffset!(ret, symbolind(ind, op, td), offset)
        end
    end
    ret
end
function offset_refname(op::Operation, td::UnrollArgs)
    rn = refname(op)
    use_gesp(op, td) ? Symbol(refname(op), "#offset#", name(op)) : rn
end
use_gesp(op::Operation, td::UnrollArgs) = all(op.ref.loopedindex)# && td.vectorized ∈ loopdependencies(op)
function gesp_call!(q::Expr, op::Operation, td::UnrollArgs)
    ref = refname(op)
    # ref_offset = offset_refname(op)
    ref_offset = Symbol(refname(op), "#offset#", name(op))
    mo = mem_offset(op, td, false, false) # false, to say emit-all-indices
    push!(q.args, Expr(:(=), ref_offset, Expr(:call, lv(:gesp), ref, mo)))
    nothing
end
function maybegesp_call!(q::Expr, op::Operation, td::UnrollArgs)
    @unpack u₁loopsym, u₂loopsym, suffix = td
    if ((isnothing(suffix) || iszero(suffix)) && ((u₁loopsym ∈ loopdependencies(op)) || (u₂loopsym ∈ loopdependencies(op))) && use_gesp(op, td))
        gesp_call!(q, op, td)
    end
end


function add_vectorized_offset!(ret::Expr, ind, offset, incr)
    if isone(incr)
        if iszero(offset)
            push!(ret.args, _MMind(Expr(:call, lv(:valadd), VECTORWIDTHSYMBOL, ind)))
        else
            push!(ret.args, _MMind(Expr(:call, lv(:vadd), ind, Expr(:call, lv(:valadd), VECTORWIDTHSYMBOL, convert(Int, offset)))))
        end
    elseif iszero(incr)
        if iszero(offset)
            push!(ret.args, _MMind(ind))
        else
            addoffset!(ret, ind, offset, true)
        end
    elseif iszero(offset)
        push!(ret.args, _MMind(Expr(:call, lv(:valmuladd), VECTORWIDTHSYMBOL, incr, ind)))
    else
        push!(ret.args, _MMind(Expr(:call, lv(:vadd), ind, Expr(:call, lv(:valmuladd), VECTORWIDTHSYMBOL, incr, convert(Int, offset)))))
    end
end
function add_vectorized_offset_unrolled!(ret::Expr, offset, incr)
    if isone(incr)
        if iszero(offset)
            push!(ret.args, _MMind(Expr(:call, lv(:unwrap), VECTORWIDTHSYMBOL)))
        else
            push!(ret.args, _MMind(Expr(:call, lv(:valadd), VECTORWIDTHSYMBOL, convert(Int, offset))))
        end
    elseif iszero(incr)
        if iszero(offset)
            push!(ret.args, _MMind(Expr(:call, lv(:Zero))))
        else
            push!(ret.args, _MMind(convert(Int, offset)))
        end
    elseif iszero(offset)
        push!(ret.args, _MMind(Expr(:call, lv(:valmul), VECTORWIDTHSYMBOL, incr)))
    else
        push!(ret.args, _MMind(Expr(:call, lv(:valmuladd), VECTORWIDTHSYMBOL, incr, convert(Int, offset))))
    end
end
function add_vectorized_offset!(ret::Expr, ind, offset, incr, unrolled)
    if unrolled
        add_vectorized_offset_unrolled!(ret, offset, incr)
    else
        add_vectorized_offset!(ret, ind, offset, incr)
    end
end
function mem_offset_u(op::Operation, td::UnrollArgs, unrolled::Bool)
    @assert accesses_memory(op) "Computing memory offset only makes sense for operations that access memory."
    @unpack u₁, u₁loopsym, u₂loopsym, vectorized, suffix = td
    incr₁ = u₁
    incr₂ = isnothing(suffix) ? 0 : suffix
    ret = Expr(:tuple)
    indices = getindicesonly(op)
    offsets = getoffsets(op)
    loopedindex = op.ref.loopedindex
    unrolled = unrolled && use_gesp(op, td)
    if iszero(incr₁) & iszero(incr₂)
        return mem_offset(op, td, unrolled)
        # append_inds!(ret, indices, loopedindex)
    else
        for (n,ind) ∈ enumerate(indices)
            offset = convert(Int, offsets[n])
            # if ind isa Int # impossible
                # push!(ret.args, ind + offset)
            # else
            indvectorized = ind === vectorized
            if ind === u₁loopsym
                if indvectorized
                    add_vectorized_offset!(ret, ind, offset, incr₁, unrolled)
                elseif unrolled
                    addoffset!(ret, incr₁ + offset)
                else
                    addoffset!(ret, ind, incr₁ + offset)
                end
            elseif ind === u₂loopsym
                if indvectorized
                    add_vectorized_offset!(ret, ind, offset, incr₂, unrolled)
                elseif unrolled
                    addoffset!(ret, incr₂ + offset)
                else
                    addoffset!(ret, ind, incr₂ + offset)
                end
            elseif loopedindex[n]
                if unrolled
                    addoffset!(ret, offset, indvectorized)
                else
                    addoffset!(ret, ind, offset, indvectorized)
                end
            else
                addoffset!(ret, symbolind(ind, op, td), offset)
            end
        end
    end
    ret
end

function varassignname(var::Symbol, u::Int, isunrolled::Bool)
    isunrolled ? Symbol(var, u) : var
end
# name_memoffset only gets called when vectorized
function name_memoffset(var::Symbol, op::Operation, td::UnrollArgs, u₁unrolled::Bool, unrolled::Bool)
    @unpack u₁, u₁loopsym, u₂loopsym, suffix = td
    if isnothing(suffix) && u₁ < 0 # sentinel value meaning not unrolled
        name = var
        mo = mem_offset(op, td, unrolled)
    else
        name = u₁unrolled ? Symbol(var, u₁) : var
        mo = mem_offset_u(op, td, unrolled)
    end
    name, mo
end

function condvarname_and_unroll(cond, u₁loop, u₂loop, suffix, opu₂)
    if isnothing(suffix) || !opu₂
        condvar, condu₁, condu₂ = variable_name_and_unrolled(cond, u₁loop, u₂loop, nothing)
    else
        condvar, condu₁, condu₂ = variable_name_and_unrolled(cond, u₁loop, u₂loop, suffix)
    end
    condvar, condu₁
end

