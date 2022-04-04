
# This is a convenience function for libraries like MaBLAS and PaddedMatrices to use explicitly, and for others to force more type inference in precompilation.

const GEMMLOOPSET = loopset(:(
  for n ∈ 1:N, m ∈ 1:M
    Cₘₙ = zero(eltype(C))
    for k ∈ 1:K
      Cₘₙ += A[m, k] * B[k, n]
    end
    C[m, n] += Cₘₙ
  end
));


# function matmul_params(rs::Int, rc::Int, cls::Int)
#     set_hw!(GEMMLOOPSET, rs, rc, cls, Int(cache_size(StaticInt(1))), Int(cache_size(StaticInt(2))), Int(cache_size(StaticInt(3))))
#     order = choose_order(GEMMLOOPSET)
#     order[5], last(order)
# end
function matmul_params(
  rs::Int,
  rc::Int,
  cls::Int;
  M = nothing,
  K = nothing,
  N = nothing,
  W = 0,
)
  set_hw!(
    GEMMLOOPSET,
    rs,
    rc,
    cls,
    Int(cache_size(StaticInt(1))),
    Int(cache_size(StaticInt(2))),
    Int(cache_size(StaticInt(3))),
  )
  if N ≢ nothing
    nloop = GEMMLOOPSET.loops[1]
    GEMMLOOPSET.loops[1] =
      Loop(:n, MaybeKnown(1), MaybeKnown(N), MaybeKnown(1), nloop.rangesym, nloop.lensym)
  end
  if M ≢ nothing
    mloop = GEMMLOOPSET.loops[2]
    GEMMLOOPSET.loops[2] =
      Loop(:m, MaybeKnown(1), MaybeKnown(M), MaybeKnown(1), mloop.rangesym, mloop.lensym)
  end
  if K ≢ nothing
    kloop = GEMMLOOPSET.loops[3]
    GEMMLOOPSET.loops[3] =
      Loop(:k, MaybeKnown(1), MaybeKnown(K), MaybeKnown(1), kloop.rangesym, kloop.lensym)
  end
  GEMMLOOPSET.vector_width = W
  order = choose_order(GEMMLOOPSET)
  (N ≢ nothing) && (GEMMLOOPSET.loops[1] = nloop)
  (M ≢ nothing) && (GEMMLOOPSET.loops[2] = mloop)
  (K ≢ nothing) && (GEMMLOOPSET.loops[3] = kloop)
  order[5], last(order)
end
@generated function matmul_params(
  ::StaticInt{RS},
  ::StaticInt{RC},
  ::StaticInt{CLS},
) where {RS,RC,CLS}
  mᵣ, nᵣ = matmul_params(RS, RC, CLS)
  Expr(
    :tuple,
    Expr(:call, Expr(:curly, :StaticInt, mᵣ)),
    Expr(:call, Expr(:curly, :StaticInt, nᵣ)),
  )
end
matmul_params() = matmul_params(register_size(), register_count(), cache_linesize())

# function dotturbo(x,y)
#   s = zero(promote_type(eltype(x),eltype(y)))
#   @turbo for i ∈ eachindex(x,y)
#     s += x[i]*y[i]
#   end
#   s
# end
# dotturbo(Float64[1.0], Float64[2.0])
