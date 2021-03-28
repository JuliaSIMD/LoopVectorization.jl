using LoopVectorization, Test#, OffsetArrays

@testset "Array Wrappers" begin
	@show @__LINE__

    function addone!(y, x)
        @avx for i ∈ eachindex(x,y)
            y[i] = x[i] + 1
        end
        y
    end

    x5 = PermutedDimsArray(rand(10,10), (2,1));
    y5 = PermutedDimsArray(rand(10,10), (2,1));
    y5t = x5 .+ 1;
    @test LoopVectorization.check_args(x5)
    @test y5t == addone!(y5, x5)

    x6 = view(x5, 1:3, 1:3);
    y6 = view(y5, 1:3, 1:3);
    fill!(y6, NaN);
    @test LoopVectorization.check_args(x6) 
    addone!(y6, x6); # also 
    @test y5t == y5 # Test that `NaN`s replaced with correct answers, and that other values were untouched.

    A = rand(12,13,14,15);
    pA = PermutedDimsArray(A, (3,1,4,2));
    @test addone!(similar(pA), pA) == pA .+ 1
    ppA = PermutedDimsArray(pA, (4,2,3,1));
    @test LoopVectorization.check_args(ppA)
    @test addone!(similar(ppA), ppA) == ppA .+ 1

    x = rand(10,10); xc = copy(x);
    xv = view(x, 1:2:10, 1:3);
    xcv = view(xc, 1:2:10, 1:3);
    @test LoopVectorization.check_args(xv)
    addone!(xv, xv);
    xcv .+= 1;
    @test x == xc
end


