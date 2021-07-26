
function lv_turbo(r::Vector{UInt64}, mask::UInt64)
  @turbo for i in 1:length(r)
    r[i] &= mask
  end
  r
end

@testset "Simple Miscellaneous" begin
  @show @__LINE__
  r1 = rand(UInt, 239);
  mask = ~(one(eltype(r1))<<(2))
  r2 = r1 .& mask;
  @test lv_turbo(r1, mask) == r2;
end

