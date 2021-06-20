
@testset "Many Loop Reductions" begin
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
end

