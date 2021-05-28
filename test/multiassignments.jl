using LoopVectorization, Test


function multiassign!(y, x)
  @assert length(y)+3 == length(x)
  @inbounds for i ∈ eachindex(y)
    x₁, ((x₂,x₃), (x₄,x₅)) = x[i], (sincos(x[i+1]), (x[i+2], x[i+3]))
    y[i] = x₁ * x₄ - x₂ * x₃
  end
  y
end
multiassign(x) = multiassign!(similar(x, length(x)-3), x)
function multiassign_turbo!(y, x)
  @assert length(y)+3 == length(x)
  @turbo for i ∈ eachindex(y)
    x₁, ((x₂,x₃), (x₄,x₅)) = x[i], (sincos(x[i+1]), (x[i+2], x[i+3]))
    y[i] = x₁ * x₄ - x₂ * x₃
  end
  y
end
multiassign_turbo(x) = multiassign_turbo!(similar(x, length(x)-3), x)

@testset "Multiple assignments" begin
  x = rand(111);
  @test multiassign(x) ≈ multiassign_turbo(x)  
end

