using LoopVectorization
using Test

@testset "Utilities" begin
    @test LoopVectorization.isscopedname(:(Base.OneTo), :Base, :OneTo)
    @test LoopVectorization.isscopedname(:(A.B.C.D), (:A, :B, :C), :D)
    @test !LoopVectorization.isscopedname(:(A.B.D),  (:A, :B, :C), :D)
    @test !LoopVectorization.isscopedname(:(A.B.C.D), (:A, :B, :C), :E)
    @test !LoopVectorization.isscopedname(:hello,  :Base, :OneTo)

    @test LoopVectorization.isglobalref(GlobalRef(Base, :getindex), Base, :getindex)
    @test !LoopVectorization.isglobalref(GlobalRef(Base, :getindex), Base, :setindex!)
    @test !LoopVectorization.isglobalref(GlobalRef(Core, :getindex), Base, :getindex)
    @test !LoopVectorization.isglobalref(:getindex, Base, :getindex)
end
