@testset "map" begin
  @inline foo(x, y) = exp(x) - sin(y)
  for T ∈ (Float32,Float64)
    @show T, @__LINE__
    for N ∈ [ 3, 371 ]
      a = rand(T, N); b = rand(T, N);
      c1 = map(foo, a, b);
      c2 = vmap(foo, a, b);
      @test c1 ≈ c2
      c2 = vmapt(foo, a, b);
      @test c1 ≈ c2
      c2 = vmapnt(foo, a, b);
      @test c1 ≈ c2
      fill!(c2, NaN); @views vmapnt!(foo, c2[2:end], a[2:end], b[2:end]);
      @test @views c1[2:end] ≈ c2[2:end]
      # sleep(1e-3) # non-temporal stores won't be automatically synced/coherant, so need to wait!
      c0 = vmapntt(foo, a, b);
      c3 = similar(c0); # not aligned
      fill!(c3, NaN); @views vmapntt!(foo, c3[2:end], a[2:end], b[2:end]);
      @test c0 ≈ c1
      @test isnan(c3[begin])
      @test @views c1[2:end] ≈ c3[2:end]
    end
    
    c = rand(T,100); x = rand(T,10^4); y1 = similar(x); y2 = similar(x);
    map!(xᵢ -> clenshaw(xᵢ, c), y1, x)
    vmap!(xᵢ -> clenshaw(xᵢ, c), y2, x)
    @test y1 ≈ y2
  end
  @test vmap(abs2, 1:100) == map(abs2, 1:100)
  @test vmapt(abs2, 1:3:10000) == map(abs2, 1:3:10000)
  @test vmapt(abs2, 1.0:3.0:10000.0) ≈ map(abs2, 1.0:3.0:10000.0)
end
