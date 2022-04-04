using LoopVectorization, Test


function multiassign!(y, x)
  @assert length(y) + 3 == length(x)
  @inbounds for i ∈ eachindex(y)
    x₁, ((x₂, x₃), (x₄, x₅)) = x[i], (sincos(x[i+1]), (x[i+2], x[i+3]))
    y[i] = x₁ * x₄ - x₂ * x₃
  end
  y
end
multiassign(x) = multiassign!(similar(x, length(x) - 3), x)
function multiassign_turbo!(y, x)
  @assert length(y) + 3 == length(x)
  @turbo for i ∈ eachindex(y)
    x₁, ((x₂, x₃), (x₄, x₅)) = x[i], (sincos(x[i+1]), (x[i+2], x[i+3]))
    y[i] = x₁ * x₄ - x₂ * x₃
  end
  y
end
multiassign_turbo(x) = multiassign_turbo!(similar(x, length(x) - 3), x)

multistorefunc(x) = exp(x), sincos(x)
function multistore!(x, y, z, a)
  @inbounds for i ∈ eachindex(x, y, z, a)
    x[i], (y[i], z[i]) = multistorefunc(a[i])
  end
end
function multistore_turbo!(x, y, z, a)
  @turbo for i ∈ eachindex(x, y, z, a)
    x[i], (y[i], z[i]) = multistorefunc(a[i])
  end
end

@testset "Multiple assignments" begin
  @show @__LINE__
  x = rand(111)
  @test multiassign(x) ≈ multiassign_turbo(x)
  a0 = similar(x)
  b0 = similar(x)
  c0 = similar(x)
  a1 = similar(x)
  b1 = similar(x)
  c1 = similar(x)
  multistore!(a0, b0, c0, x)
  multistore_turbo!(a1, b1, c1, x)
  @test a0 ≈ a1
  @test b0 ≈ b1
  @test c0 ≈ c1
end
