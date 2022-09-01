module LVUser

export filter2davx

using LoopVectorization
using SnoopPrecompile

function filter2davx!(out::AbstractMatrix, A::AbstractMatrix, kern::AbstractMatrix)
    @turbo for J in CartesianIndices(out)
        tmp = zero(eltype(out))
        for I ∈ CartesianIndices(kern)
            tmp += A[I + J - 1] * kern[I]
        end
        out[J] = tmp
    end
    out
end

function filter2davx(A::AbstractMatrix, kern::AbstractMatrix)
    out = similar(A, size(A) .- size(kern) .+ 1)
    return filter2davx!(out, A, kern)
end

# precompilation
let
    A = rand(Float64, 512, 512)
    kern = [0.1 0.3 0.1;
            0.3 0.5 0.3;
            0.1 0.3 0.1]
    @precompile_all_calls begin
        filter2davx(A, kern)
    end
end

end # module LVUser
