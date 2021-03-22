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
    
end


