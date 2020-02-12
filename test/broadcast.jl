@testset "broadcast" begin
    M, N = 37, 47
    # M = 77;
    for T ∈ (Float32, Float64, Int32, Int64)
        @show T, @__LINE__
        R = T <: Integer ? (T(-100):T(100)) : T
        a = rand(R,100,100,100);
        b = rand(R,100,100,1);
        bl = LowDimArray{(true,true,false)}(b);
        br = reshape(b, (100,100));
        c1 = a .+ b;
        c2 = @avx a .+ bl;
        @test c1 ≈ c2
        fill!(c2, 99999); @avx c2 .= a .+ br;
        @test c1 ≈ c2
        fill!(c2, 99999); @avx c2 .= a .+ b;
        @test c1 ≈ c2
        br = reshape(b, (100,1,100));
        bl = LowDimArray{(true,false,true)}(br);
        @. c1 = a + br;
        fill!(c2, 99999); @avx @. c2 = a + bl;
        @test c1 ≈ c2
        fill!(c2, 99999); @avx @. c2 = a + br;
        @test c1 ≈ c2
        br = reshape(b, (1,100,100));
        bl = LowDimArray{(false,true,true)}(br);
        @. c1 = a + br;
        fill!(c2, 99999);
        @avx @. c2 = a + bl;
        @test c1 ≈ c2
        
        a = rand(R, M); B = rand(R, M, N); c = rand(R, N); c′ = c';
        d1 =      @. a + B * c′;
        d2 = @avx @. a + B * c′;
        @test d1 ≈ d2
        
        @.      d1 = a + B * c′;
        @avx @. d2 = a + B * c′;
        @test d1 ≈ d2

        d3 = a .+ B * c;
        d4 = @avx a .+ B *ˡ c;
        @test d3 ≈ d4

        fill!(d3, -1000.0);
        fill!(d4, 91000.0);

        d3 .= a .+ B * c;
        @avx d4 .= a .+ B *ˡ c;
        @test d3 ≈ d4

        fill!(d4, 91000.0);
        @avx @. d4 = a + B *ˡ c;
        @test d3 ≈ d4

        M, K, N = 77, 83, 57;
        A = rand(R,M,K); B = rand(R,K,N); C = rand(R,M,N);
        At = copy(A')
        D1 = C .+ A * B;
        D2 = @avx C .+ A *ˡ B;
        @test D1 ≈ D2
        fill!(D2, -999999); D2 = @avx C .+ At' *ˡ B;
        @test D1 ≈ D2

        b = rand(T,K); x = rand(R,N);
        D1 .= C .+ A * (b .+ x');
        @avx @. D2 = C + A *ˡ (b + x');
        @test D1 ≈ D2
        D2 = @avx @. C + A *ˡ (b + x');
        @test D1 ≈ D2        
        
        if T <: Union{Float32,Float64}
            D3 = cos.(B');
            D4 = @avx cos.(B');
            @test D3 ≈ D4
            
            fill!(D3, -1e3); fill!(D4, 9e9);
            Bt = Transpose(B);
            @. D3 = exp(Bt);
            @avx @. D4 = exp(Bt);
            @test D3 ≈ D4

            D1 = similar(B); D2 = similar(B);
            D1t = Transpose(D1);
            D2t = Transpose(D2);
            @. D1t = exp(Bt);
            @avx @. D2t = exp(Bt);
            @test D1t ≈ D2t

            fill!(D1, -1e3);
            fill!(D2, 9e9);
            @. D1' = exp(Bt);
            lset = @avx @. D2' = exp(Bt);
            
            @test D1 ≈ D2

            a = rand(137);
            b1 = @avx @. 3*a + sin(a) + sqrt(a);
            b2 =      @. 3*a + sin(a) + sqrt(a);
            @test b1 ≈ b2
            three = 3; fill!(b1, -9999);
            @avx @. b1 = three*a + sin(a) + sqrt(a);
            @test b1 ≈ b2

            C = rand(100,10,10);
            D1 = C .^ 0.3;
            D2 = @avx C .^ 0.3;
            @test D1 ≈ D2
        end
    end
end
