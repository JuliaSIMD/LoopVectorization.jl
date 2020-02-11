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
end
