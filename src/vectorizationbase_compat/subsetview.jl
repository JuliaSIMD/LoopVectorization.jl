
@generated function subsetview(
    ptr::StridedPointer{T,N,C,B,R,X,O}, ::Val{I}, i::Integer
) where {T,N,C,B,R,X,O,I}
    I > N && return :ptr
    @assert B ≤ 0 "Batched dims not currently supported."
    newC = C == I ? -1 : (C < I ? C : C - 1)
    newR = Expr(:tuple)
    newstrd = Expr(:tuple)
    newoffsets = Expr(:tuple)
    ind = Expr(:tuple)
    for i ∈ 1:I-1
        push!(newR.args, R[i])
        push!(newstrd.args, :(strd[$i]))
        push!(newoffsets.args, :(offsets[$i]))
        push!(ind.args, :(offsets[$i]))
        # push!(ind.args, Expr(:call, lv(:Zero)))
    end
    push!(ind.args, :i)
    for i ∈ I+1:N
        push!(newR.args, R[i])
        push!(newstrd.args, :(strd[$i]))
        push!(newoffsets.args, :(offsets[$i]))
        push!(ind.args, :(offsets[$i]))
        # push!(ind.args, Expr(:call, lv(:Zero)))
    end
    gptr = Expr(:call, :gep, :ptr, ind)
    quote
        $(Expr(:meta,:inline))
        strd = ptr.strd
        offsets = ptr.offsets
        StridedPointer{$T,$(N-1),$newC,$B,$newR}($gptr, $newstrd, $newoffsets)
    end
end

