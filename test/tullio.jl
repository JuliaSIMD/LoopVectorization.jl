using LoopVectorization, Test
# Tests for Tullio

@testset "Tullio Tests" begin
    A = (1:10) .^ 2; K = rand(10);

    function act!(ℛ::AbstractArray, A, 𝒶𝓍i = axes(A,1), 𝒶𝓍j = axes(ℛ,2))
        @avx for j in 𝒶𝓍j
            for i in 𝒶𝓍i
                ℛ[i, j] = A[i] / j
            end
        end
        ℛ
    end
    function act_noavx!(ℛ::AbstractArray, A, 𝒶𝓍i = axes(A,1), 𝒶𝓍j = axes(ℛ,2))
        for j in 𝒶𝓍j
            for i in 𝒶𝓍i
                ℛ[i, j] = A[i] / j
            end
        end
        ℛ
    end
    @test act!(rand(10,10), A) ≈ act_noavx!(rand(10,10), A)

    D = similar(A, 10, 10) .= 999;

    inds = [2,3,5,2];

    function two!(ℛ::AbstractArray, inds, A, 𝒶𝓍j = axes(ℛ,2), 𝒶𝓍i = axes(inds,1))         
        @avx for i = 𝒶𝓍i
            for j = 𝒶𝓍j
                ℛ[inds[i], j] = A[j]
            end
        end
        ℛ
    end
    function two_noavx!(ℛ::AbstractArray, inds, A, 𝒶𝓍j = axes(ℛ,2), 𝒶𝓍i = axes(inds,1))         
        for i = 𝒶𝓍i
            for j = 𝒶𝓍j
                ℛ[inds[i], j] = A[j]
            end
        end
        ℛ
    end
    @test two!(copy(D), inds, A) == two!(copy(D), inds, A)

    function three!(ℛ::AbstractArray, A, 𝒶𝓍i = axes(ℛ,1))
        @avx for i = 𝒶𝓍i
            ℛ[i] = A[2i + 1] + A[i]
        end
        ℛ
    end
    function three_noavx!(ℛ::AbstractArray, A, 𝒶𝓍i = axes(ℛ,1))
        for i = 𝒶𝓍i
            ℛ[i] = A[2i + 1] + A[i]
        end
        ℛ
    end
    @test three!(rand(4), A) == three_noavx!(rand(4), A)

    function and(A, 𝒶𝓍i = axes(A,1))
        𝒜𝒸𝒸 = true
        @avx for i = 𝒶𝓍i
            𝒜𝒸𝒸 = 𝒜𝒸𝒸 & (A[i] > 0)
        end
        𝒜𝒸𝒸
    end
    @test and(A)
    A[3] = -1
    @test !and(A)


    function dadada!(EV, Fu, V, ♻️ = nothing)
        (ndims)(EV) == 5 || (throw)("expected a 5-array EV")
        (ndims)(Fu) == 2 || (throw)("expected a 2-array Fu")
        (ndims)(V) == 4 || (throw)("expected a 4-array V")

        local 𝒶𝓍a = (axes)(EV, 1)
        local 𝒶𝓍k = (axes)(EV, 2)
        (axes)(V, 1) == (axes)(EV, 2) || (throw)("range of index k must agree")
        local 𝒶𝓍iu = (axes)(Fu, 2)
        (axes)(V, 4) == (axes)(Fu, 2) || (throw)("range of index iu must agree")
        local 𝒶𝓍z = (axes)(EV, 4)
        (axes)(V, 3) == (axes)(EV, 4) || (throw)("range of index z must agree")
        local 𝒶𝓍u = (axes)(EV, 5)
        (axes)(Fu, 1) == (axes)(EV, 5) || (throw)("range of index u must agree")
        local 𝒶𝓍x = (axes)(EV, 3)
        (axes)(V, 2) == (axes)(EV, 3) || (throw)("range of index x must agree")

        ℛ = EV

        @avx for u = 𝒶𝓍u
            for z = 𝒶𝓍z
                for x = 𝒶𝓍x
                    for k = 𝒶𝓍k
                        for a = 𝒶𝓍a
                            𝒜𝒸𝒸 = zero(eltype(EV))  # simpler, same error
                            # 𝒜𝒸𝒸 = if ♻️ === nothing
                            #         zero(𝒯)
                            #     else
                            #         ℛ[a, k, x, z, u]
                            #     end
                            for iu = 𝒶𝓍iu
                                𝒜𝒸𝒸 = 𝒜𝒸𝒸 + Fu[u, iu] * V[k, x, z, iu]
                            end
                            ℛ[a, k, x, z, u] = 𝒜𝒸𝒸
                        end
                    end
                end
            end
        end
    end
    function dadada_noavx!(EV, Fu, V, ♻️ = nothing)
        (ndims)(EV) == 5 || (throw)("expected a 5-array EV")
        (ndims)(Fu) == 2 || (throw)("expected a 2-array Fu")
        (ndims)(V) == 4 || (throw)("expected a 4-array V")

        local 𝒶𝓍a = (axes)(EV, 1)
        local 𝒶𝓍k = (axes)(EV, 2)
        (axes)(V, 1) == (axes)(EV, 2) || (throw)("range of index k must agree")
        local 𝒶𝓍iu = (axes)(Fu, 2)
        (axes)(V, 4) == (axes)(Fu, 2) || (throw)("range of index iu must agree")
        local 𝒶𝓍z = (axes)(EV, 4)
        (axes)(V, 3) == (axes)(EV, 4) || (throw)("range of index z must agree")
        local 𝒶𝓍u = (axes)(EV, 5)
        (axes)(Fu, 1) == (axes)(EV, 5) || (throw)("range of index u must agree")
        local 𝒶𝓍x = (axes)(EV, 3)
        (axes)(V, 2) == (axes)(EV, 3) || (throw)("range of index x must agree")

        ℛ = EV

        @inbounds @fastmath for u = 𝒶𝓍u
            for z = 𝒶𝓍z
                for x = 𝒶𝓍x
                    for k = 𝒶𝓍k
                        for a = 𝒶𝓍a
                            𝒜𝒸𝒸 = zero(eltype(EV))  # simpler, same error
                            # 𝒜𝒸𝒸 = if ♻️ === nothing
                            #         zero(𝒯)
                            #     else
                            #         ℛ[a, k, x, z, u]
                            #     end
                            for iu = 𝒶𝓍iu
                                𝒜𝒸𝒸 = 𝒜𝒸𝒸 + Fu[u, iu] * V[k, x, z, iu]
                            end
                            ℛ[a, k, x, z, u] = 𝒜𝒸𝒸
                        end
                    end
                end
            end
        end
    end
    EV, Fu, V = rand(3,3,3,3,3), rand(3,3), rand(3,3,3,3);
    EV2 = similar(EV);
    dadada!(EV, Fu, V)
    dadada_noavx!(EV2, Fu, V)
    @test EV ≈ EV2
end


