@testset "Print Methods" begin
    @show @__LINE__
    selfdotq = :(for i ∈ eachindex(a)
                 s += a[i]*a[i]
                 end)
    lsselfdot = LoopVectorization.LoopSet(selfdotq);
    io = IOBuffer();
    println(io, LoopVectorization.operations(lsselfdot))
    s = String(take!(io))
    @test occursin("Operation[", s)
    @test occursin("s = 0", s)
    @test occursin("s = LoopVectorization.vfmadd", s)

    AmulBq = :(for m ∈ 1:M, n ∈ 1:N
                   C[m,n] = zero(eltype(B))
                   for k ∈ 1:K
                       C[m,n] += A[m,k] * B[k,n]
                   end
               end);
    lsAmulB = LoopVectorization.LoopSet(AmulBq);
    println(io, LoopVectorization.operations(lsAmulB))
    s = String(take!(io))
    @test occursin("Operation[", s)
    @test occursin("C[m, n] = ", s)
    @test occursin(" = A[m, k]", s)
    @test occursin(" = B[k, n]", s)
    @test occursin(" = LoopVectorization.vfmadd", s)
    @test occursin(" = LoopVectorization.reduce_to_add", s)
end
