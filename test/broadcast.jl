using LoopVectorization, Test
# T = Float32

macro outline(ex)
  quote
    (() -> begin
       Base.@_noinline_meta
       $(esc(ex))
     end)()
  end
end

function test_broadcast(::Type{T}) where {T}
  M, N = 37, 47
  @show T, @__LINE__
  R = T <: Integer ? (T(-100):T(100)) : T
  a = rand(R,99,99,99);
  b = rand(R,99,99,1);
  bl = LowDimArray{(true,true,false)}(b);
  @test size(bl) == size(b)
  @test LoopVectorization.ArrayInterface.size(bl) === (size(b,1),size(b,2),LoopVectorization.StaticInt(1))

  br = reshape(b, (99,99));
  c1 = a .+ b;
  c2 = @outline @turbo a .+ bl;
  @test c1 ≈ c2
  fill!(c2, 99999); @outline @turbo c2 .= a .+ br;
  @test c1 ≈ c2
  fill!(c2, 99999); @outline @turbo c2 .= a .+ b;
  @test c1 ≈ c2
  br = reshape(b, (99,1,99));
  bl = LowDimArray{(true,false,true)}(br);
  @test size(bl) == size(br)
  @test LoopVectorization.ArrayInterface.size(bl) === (size(br,1),LoopVectorization.StaticInt(1),size(br,3))
  @. c1 = a + br;
  fill!(c2, 99999); @outline @turbo @. c2 = a + bl;
  @test c1 ≈ c2
  fill!(c2, 99999); @outline @turbo @. c2 = a + br;
  @test c1 ≈ c2
  br = reshape(b, (1,99,99));
  bl = LowDimArray{(false,)}(br);
  @test size(bl) == size(br)
  @test LoopVectorization.ArrayInterface.size(bl) === (LoopVectorization.StaticInt(1),size(br,2),size(br,3))
  @. c1 = a + br;
  fill!(c2, 99999);
  @test c1 ≈ @outline @turbo @. c2 = a + bl
  # @test c1 ≈ c2
  br = reshape(rand(R,99), (1,99,1));
  bl = LowDimArray{(false,)}(br);
  @test size(bl) == size(br)
  @. c1 = a + br;
  fill!(c2, 99999);
  @outline @turbo @. c2 = a + bl;
  @test c1 ≈ c2

  if T <: Integer
    xs = rand(-T(100):T(100), M);
  else
    xs = rand(T, M);
  end
  max_ = maximum(xs, dims=1);
  @test (@outline @turbo exp.(xs .- LowDimArray{(false,)}(max_))) ≈ exp.(xs .- LowDimArray{(false,)}(max_))
  @test size(LowDimArray{(false,)}(max_)) == size(max_)

  if T === Int32
    a = rand(T(1):T(100), 73, 1);
    @test sqrt.(Float32.(a)) ≈ @outline @turbo sqrt.(a)
  elseif T === Int64
    a = rand(T(1):T(100), 73, 1);
    @test sqrt.(a) ≈ @outline @tturbo sqrt.(a)
  else
    a = rand(T, 73, 1);
    @test sqrt.(a) ≈ @outline @turbo sqrt.(a)
  end
  
  a = rand(R, M); B = rand(R, M, N); c = rand(R, N); c′ = c';
  d1 =       @. a + B * c′;
  d2 = @outline @tturbo @. a + B * c′;
  @test d1 ≈ d2
  
  @.      d1 = a + B * c′;
  @outline @turbo @. d2 = a + B * c′;
  @test d1 ≈ d2

  d3 = a .+ B * c;
  d4 = @outline @turbo a .+ B *ˡ c;
  @test d3 ≈ d4

  fill!(d3, -1000.0);
  fill!(d4, 91000.0);

  d3 .= a .+ B * c;
  @outline @turbo d4 .= a .+ B *ˡ c;
  @test d3 ≈ d4

  fill!(d4, 91000.0);
  @outline @turbo @. d4 = a + B *ˡ c;
  @test d3 ≈ d4

  M, K, N = 77, 83, 57;
  A = rand(R,M,K); B = rand(R,K,N); C = rand(R,M,N);
  At = copy(A');
  D1 = C .+ A * B;
  D2 = @outline @tturbo C .+ A .*ˡ B;
  @test D1 ≈ D2
  if RUN_SLOW_TESTS
    fill!(D2, -999999); D2 = @outline @turbo C .+ At' *ˡ B;
    @test D1 ≈ D2
    fill!(D2, -999999); @test A * B ≈ (@outline @turbo @. D2 = A *ˡ B)
    D1 .= view(C, 1, :)' .+ A * B;
    fill!(D2, -999999);
    @outline @turbo D2 .= view(C, 1, :)' .+ A .*ˡ B;
    @test D1 ≈ D2
    C3d = rand(R,3,M,N);
    D1 .= view(C3d, 1, :, :) .+ A * B;
    fill!(D2, -999999);
    @outline @tturbo D2 .= view(C3d, 1, :, :) .+ A .*ˡ B;
    @test D1 ≈ D2
  end
  D1 .= 9999;
  @outline @turbo D2 .= 9999;
  @test D1 == D2
  D1 .= -99999;
  @outline @tturbo D2' .= -99999;
  @test D1 == D2
  
  b = rand(T,K); x = rand(R,N);
  D1 .= C .+ A * (b .+ x');
  @outline @tturbo @. D2 = C + A *ˡ (b + x');
  @test D1 ≈ D2
  D2 = @outline @turbo @. C + A *ˡ (b + x');
  @test D1 ≈ D2
  
  if T <: Union{Float32,Float64}
    D3 = cos.(B');
    D4 = @outline @turbo cos.(B');
    @test D3 ≈ D4
    
    fill!(D3, -1e3); fill!(D4, 9e9);
    Bt = transpose(B);
    @. D3 = exp(Bt);
    @outline @tturbo @. D4 = exp(Bt);
    @test D3 ≈ D4

    D1 = similar(B); D2 = similar(B);
    D1t = transpose(D1);
    D2t = transpose(D2);
    @. D1t = exp(Bt);
    @outline @turbo @. D2t = exp(Bt);
    @test D1t ≈ D2t

    fill!(D1, -1e3);
    fill!(D2, 9e9);
    @. D1' = exp(Bt);
    lset = @outline @tturbo @. D2' = exp(Bt);
    
    @test D1 ≈ D2

    a = rand(137);
    b1 = @outline @turbo @. 3*a + sin(a) + sqrt(a);
    b2 =      @. 3*a + sin(a) + sqrt(a);
    @test b1 ≈ b2
    three = 3; fill!(b1, -9999);
    @outline @tturbo @. b1 = three*a + sin(a) + sqrt(a);
    @test b1 ≈ b2

    C = rand(100,10,10);
    D1 = C .^ 0.3;
    D2 = @outline @tturbo C .^ 0.3;
    @test D1 ≈ D2
    @. D1 = C ^ 2;
    @outline @turbo @. D2 = C ^ 2;
    @test D1 ≈ D2
    @turbo view(C,1:100,1:10,1:10) .= 0;
    @test all(==(0), C)
    @turbo view(B, axes(B,1), axes(B,2))' .= 2
    @test all(==(2), B)
  end
end

@testset "broadcast_float" begin
  for T ∈ (Float32, Float64, Int32, Int64)
    @time test_broadcast(T)
  end
end


