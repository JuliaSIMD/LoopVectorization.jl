
function mismatchedreductions_noturbo!(𝛥r392, 𝛥x923, 𝛥ℛ, ℛ, r392, x923, 𝒶𝓍k=1:2, 𝒶𝓍n=1:3, 𝒶𝓍j=1:9, 𝒶𝓍m=1:9, 𝒶𝓍i=1:3)
  @inbounds @fastmath for k = 𝒶𝓍k
    for i = 𝒶𝓍i
      for m = 𝒶𝓍m
        for j = 𝒶𝓍j
          for n = 𝒶𝓍n
            ℰ𝓍1 = conj(x923[m, k, n])
            ℰ𝓍2 = 𝛥ℛ[n, j, m, i] * ℰ𝓍1
            ℰ𝓍3 = conj(r392[i, j, k])
            ℰ𝓍4 = 𝛥ℛ[n, j, m, i] * ℰ𝓍3
            𝛥r392[i, j, k] = 𝛥r392[i, j, k] + ℰ𝓍2
            𝛥x923[m, k, n] = 𝛥x923[m, k, n] + ℰ𝓍4
          end
        end
      end
    end
  end
  𝛥r392, 𝛥x923
end
function mismatchedreductions!(𝛥r392, 𝛥x923, 𝛥ℛ, ℛ, r392, x923, 𝒶𝓍k=1:2, 𝒶𝓍n=1:3, 𝒶𝓍j=1:9, 𝒶𝓍m=1:9, 𝒶𝓍i=1:3)
  @turbo for k = 𝒶𝓍k
    for i = 𝒶𝓍i
      for m = 𝒶𝓍m
        for j = 𝒶𝓍j
          for n = 𝒶𝓍n
            ℰ𝓍1 = conj(x923[m, k, n])
            ℰ𝓍2 = 𝛥ℛ[n, j, m, i] * ℰ𝓍1
            ℰ𝓍3 = conj(r392[i, j, k])
            ℰ𝓍4 = 𝛥ℛ[n, j, m, i] * ℰ𝓍3
            𝛥r392[i, j, k] = 𝛥r392[i, j, k] + ℰ𝓍2
            𝛥x923[m, k, n] = 𝛥x923[m, k, n] + ℰ𝓍4
          end
        end
      end
    end
  end
  𝛥r392, 𝛥x923
end

@testset "Many Loop Reductions" begin
  @show @__LINE__
  A = rand((2:6)...);
  N = ndims(A)
  T = eltype(A)
  let dims = (3,5)
    sᵢ = size(A)
    sₒ = ntuple(Val(N)) do d
      ifelse(d ∈ dims, 1, sᵢ[d])
    end
    Tₒ = Base.promote_op(+, T, Int)
    B = similar(A, Tₒ, sₒ);

    Bᵥ = view(B, Colon(), Colon(), firstindex(B, 3), Colon(), firstindex(B, 5))
    @turbo for i_1 = indices((A, B), 1)
      for i_2 = indices((A, B), 2)
        for i_4 = indices((A, B), 4)
          Σ = zero(eltype(Bᵥ))
          for i_3 = axes(A, 3)
            for i_5 = axes(A, 5)
              Σ += A[i_1, i_2, i_3, i_4, i_5]
            end
          end
          Bᵥ[i_1, i_2, i_4] = Σ
        end
      end
    end
    @test B ≈ sum(A, dims = dims)
  end
  let dims = (1,2,5)

    sᵢ = size(A)
    sₒ = ntuple(Val(N)) do d
      ifelse(d ∈ dims, 1, sᵢ[d])
    end
    Tₒ = Base.promote_op(+, T, Int)
    B = similar(A, Tₒ, sₒ);

    Bᵥ = view(B, firstindex(B, 1), firstindex(B, 2), Colon(), Colon(), firstindex(B, 5))
    @turbo for i_3 = indices((A, B), 3)
      for i_4 = indices((A, B), 4)
        Σ = zero(eltype(Bᵥ))
        for i_1 = axes(A, 1)
          for i_2 = axes(A, 2)
            for i_5 = axes(A, 5)
              Σ += A[i_1, i_2, i_3, i_4, i_5]
            end
          end
        end
        Bᵥ[i_3, i_4] = Σ
      end
    end
    @test B ≈ sum(A, dims = dims)
  end

  r392 = rand(3,9,2);
  x923 = rand(9,2,3);
  K = rand(3,9,9,3);
  𝛥r392_1, 𝛥x923_1, 𝛥r392_2, 𝛥x923_2, 𝛥ℛ = similar(r392), similar(x923), similar(r392), similar(x923), copy(K);
  𝛥r392_1 .= -1; 𝛥x923_1 .= -1; 𝛥r392_2 .= -1; 𝛥x923_2 .= -1;

  mismatchedreductions_noturbo!(𝛥r392_1, 𝛥x923_1, 𝛥ℛ, K, r392, x923)
  @time mismatchedreductions!(𝛥r392_2, 𝛥x923_2, 𝛥ℛ, K, r392, x923)
  @test 𝛥r392_1 ≈ 𝛥r392_2
  @test 𝛥x923_1 ≈ 𝛥x923_2
end

