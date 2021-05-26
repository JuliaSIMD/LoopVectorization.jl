
using LoopVectorization, SnoopCompile
tinf = @snoopi_deep include(joinpath(pkgdir(LoopVectorization), "test", "runtests.jl"))

ttot, pcs = SnoopCompile.parcel(tinf);

lv, pcslv = last(pcs);
@assert lv === LoopVectorization

blacklist = (
  :_turbo_!, :vmaterialize!, :vmaterialize, Symbol("vreduce##kw"), :_vreduce_dims!, :vreduce, :vmapreduce, :SIMDMapBack, :launch_thread_vmap!, :_turbo_loopset_debug, :all_dense,
  :sigmoid_fast, :rrule, :add_broadcast!, :create_mrefs!, :avx_config_val, :subsetview, :ifelsepartial, :tanh_fast, :check_args, :relu, :init_dual
)
filteredmethods = filter(m -> !Base.sym_in(m[2].def.name, blacklist), last(pcslv)); length(filteredmethods)

SnoopCompile.write("/tmp/precompile_loopvec", [LoopVectorization => (sum(first,filteredmethods),filteredmethods)])


# pc = SnoopCompile.parcel(tinf; blacklist=["vmaterialize", "vmaterialize!", "vreduce", "Base.Broadcast.materialize", "_vreduce_dims!", "vmapreduce"])
# pcs = pc[:LoopVectorization]
# open(joinpath(pkgdir, "src", "precompile.jl"), "w") do io
#     println(io, """
#     function _precompile_()
#         ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
#     """)
#     for stmt in sort(pcs)
#         println(io, "    ", stmt)
#     end
#     println(io, "end")
# end
