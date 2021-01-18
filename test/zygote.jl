using Zygote
zygotefun2(a,b) = sigmoid_fast(a + b) + LoopVectorization.relu(a * b) * tanh(a - b)
zygotefun3(a,b,c) = log(tanh_fast(a*b) + 1 + exp(b - c))
@testset "Zygote" begin
    @show @__LINE__

    for T ∈ (Float32,Float64)
        x = randn(T, 217); y = randn(T, 217); z = randn(T, 217);
        w = randn(T, 217)
        # Test 1-arg and `tanh_fast` vs `tanh`
        gref1 = gradient(x -> dot(w,  map(tanh,      x)), x);
        glv_1 = gradient(x -> dot(w, vmap(tanh_fast, x)), x);

        @test  only(gref1) ≈ only(glv_1)

        # Test 2 arguments
        # Test 3 arguments
        gref2 = gradient(xyz -> dot(w,  map(zygotefun2, xyz...)), (x, y));
        g_lv2 = gradient(xyz -> dot(w, vmap(zygotefun2, xyz...)), (x, y));

        @test all(map(≈, only(gref2), only(g_lv2)))

        gref3 = gradient(xyz -> dot(w,  map(zygotefun3, xyz...)), (x, y, z));
        g_lv3 = gradient(xyz -> dot(w, vmap(zygotefun3, xyz...)), (x, y, z));

        @test all(map(≈, only(gref3), only(g_lv3)))
    end
end


