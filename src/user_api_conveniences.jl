
# This is a convenience function for libraries like MaBLAS and PaddedMatrices to use explicitly, and for others to force more type inference in precompilation.
let GEMMLOOPSET = LoopVectorization.LoopSet(
    :(for m ∈ 1:size(A,1), n ∈ 1:size(B,2)
          Cmn = zero(eltype(C))
          for k ∈ 1:size(A,2)
              Cmn += A[m,k] * B[k,n]
          end
          C[m,n] += Cmn
      end)
    );
    order = LoopVectorization.choose_order(GEMMLOOPSET)
    mr = order[5]
    nr = last(order)
    @eval const mᵣ = $mr
    @eval const nᵣ = $nr
end

