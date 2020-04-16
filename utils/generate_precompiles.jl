using LoopVectorization, SnoopCompile
pkgdir = dirname(@__DIR__)
tinf = @snoopi tmin=0.01 include(joinpath(pkgdir, "test", "runtests.jl"))
pc = SnoopCompile.parcel(tinf; blacklist=["vmaterialize", "vmaterialize!"])
pcs = pc[:LoopVectorization]
open(joinpath(pkgdir, "src", "precompile.jl"), "w") do io
    println(io, """
    function _precompile_()
        ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    """)
    for stmt in sort(pcs)
        println(io, "    ", stmt)
    end
    println(io, "end")
end