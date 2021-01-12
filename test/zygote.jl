using Zygote
zygotefun(a,b,c) = log(tanh_fast(a) + 1 + exp(b - c))
@testset "Zygote" begin
    @show @__LINE__

    for T ∈ (Float32,Float64)
        x = randn(T, 217); y = randn(T, 217); z = randn(T, 217);

        # Test 1-arg and `tanh_fast` vs `tanh`
        gtref = gradient(x -> sum(map(tanh, x)), x);
        gtlv = gradient(x -> sum(vmap(tanh_fast, x)), x);

        @test  only(gtref) ≈ only(gtlv)

        # Test multiple arguments
        gref = gradient(xyz -> sum(map(zygotefun, xyz...)), (x, y, z));
        glv = gradient(xyz -> sum(vmap(zygotefun, xyz...)), (x, y, z));

        @test all(map(≈, only(gref), only(glv)))
    end
end


