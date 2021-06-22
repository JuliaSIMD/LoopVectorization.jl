using OffsetArrays

@testset "Iteration Bound Tests" begin
  function masktest_incr1_none1start!(y,x)
    @turbo for i ∈ 0:20
      y[i] = x[i] + 2
    end
  end
  function masktest_incr2_none1start!(y,x)
    @turbo for i ∈ 0:2:20
      y[i] = x[i] + 2
    end
  end

  x = OffsetVector(rand(24),0:23);
  y = copy(x);
  masktest_incr1_none1start!(y,x)
  @test y == x .+ ifelse.(axes(x,1) .≤ 20, 2, 0)
  @turbo y .= x;
  @test y == x
  masktest_incr2_none1start!(y,x)
  @test y == x .+ ifelse.((axes(x,1) .≤ 20) .& iseven.(axes(x,1)), 2, 0)


  # issue #290
  function my_gemm_noturbo!(out, s::Matrix{UInt8}, V, srows, scols, Vcols, μ)
    k = srows >> 2
    rem = srows & 3
    fill!(out, 0)

    for c in 1:Vcols
      for i in 1:scols
        for l in 1:k
          block = s[l, i]
          for p in 1:4
            Aij = (block >> (2 * (p - 1))) & 3
            out[i, c] += (((Aij >= 2) + (Aij == 3) + (Aij == 1) * μ[i]) *
                          V[4 * (l - 1) + p, c])
          end
        end
      end
    end
    nothing
  end
  function my_gemm!(out, s::Matrix{UInt8}, V, srows, scols, Vcols, μ)
    k = srows >> 2
    rem = srows & 3
    fill!(out, 0)

    @avx for c in 1:Vcols
      for i in 1:scols
        for l in 1:k
          block = s[l, i]
          for p in 1:4
            Aij = (block >> (2 * (p - 1))) & 3
            out[i, c] += (((Aij >= 2) + (Aij == 3) + (Aij == 1) * μ[i]) *
                          V[4 * (l - 1) + p, c])
          end
        end
      end
    end
    nothing
  end

  out_true = Matrix{Float64}(undef, 100, 100);
  out_test1 = similar(out_true);
  # out_test2 = zeros(100, 100)
  μ = rand(100);
  s = rand(UInt8, 100, 100);
  V = rand(400, 100);

  my_gemm_noturbo!(out_true, s, V, 400, 100, 100, μ)
  my_gemm!(out_test1, s, V, 400, 100, 100, μ);
  @test out_true ≈ out_test1
end
