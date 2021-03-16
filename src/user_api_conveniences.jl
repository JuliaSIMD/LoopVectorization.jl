
# This is a convenience function for libraries like MaBLAS and PaddedMatrices to use explicitly, and for others to force more type inference in precompilation.

const GEMMLOOPSET = loopset(
    :(for n ∈ 1:N, m ∈ 1:M
          Cₘₙ = zero(eltype(C))
          for k ∈ 1:K
              Cₘₙ += A[m,k] * B[k,n]
          end
          C[m,n] += Cₘₙ
      end)
);


function matmul_params(rs::Int, rc::Int, cls::Int)
    set_hw!(GEMMLOOPSET, rs, rc, cls, Int(cache_size(StaticInt(1))), Int(cache_size(StaticInt(2))), Int(cache_size(StaticInt(3))))
    order = choose_order(GEMMLOOPSET)
    order[5], last(order)
end
@generated function matmul_params(::StaticInt{RS}, ::StaticInt{RC}, ::StaticInt{CLS}) where {RS,RC,CLS}
    mᵣ, nᵣ = matmul_params(RS, RC, CLS)
    Expr(:tuple, Expr(:call, Expr(:curly, :StaticInt, mᵣ)), Expr(:call, Expr(:curly, :StaticInt, nᵣ)))
end
matmul_params() = matmul_params(register_size(), register_count(), cache_linesize())

