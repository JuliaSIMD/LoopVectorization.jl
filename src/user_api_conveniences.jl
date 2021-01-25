
# This is a convenience function for libraries like MaBLAS and PaddedMatrices to use explicitly, and for others to force more type inference in precompilation.

@generated function matmul_params()
    gemmloopset = LoopSet(
    :(for n ∈ 1:N, m ∈ 1:M
          Cₘₙ = zero(eltype(C))
          for k ∈ 1:K
              Cₘₙ += A[m,k] * B[k,n]
          end
          C[m,n] += Cₘₙ
      end)
    )
    order = choose_order(gemmloopset)
    mᵣ = order[5]
    nᵣ = last(order)
    Expr(:tuple, Expr(:call, Expr(:curly, :StaticInt, mᵣ)), Expr(:call, Expr(:curly, :StaticInt, nᵣ)))
end

