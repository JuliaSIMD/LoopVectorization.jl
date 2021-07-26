include("testsetup.jl")

import InteractiveUtils, Aqua

InteractiveUtils.versioninfo(stdout; verbose = true)

const LOOPVECTORIZATION_TEST = get(ENV, "LOOPVECTORIZATION_TEST", "all")

if LOOPVECTORIZATION_TEST == "all"
  NUMGROUPS = 6
  processes = Vector{Base.Process}(undef, NUMGROUPS)
  paths = Vector{String}(undef, NUMGROUPS)
  ios = Vector{IOStream}(undef, NUMGROUPS)
  tmp = tempdir();
  for i ∈ 1:NUMGROUPS
    path, io = mktemp(tmp)
    paths[i] = path
    ios[i] = io
    env = copy(ENV)
    env["LOOPVECTORIZATION_TEST"] = "part$i"
    env["JULIA_NUM_THREADS"] = string(Threads.nthreads())
    processes[i] = run(pipeline(setenv(`$(Base.julia_cmd()) $(@__FILE__)`, env), stderr = io, stdout = io), wait=false)
  end
  for i ∈ 1:NUMGROUPS
    proc = processes[i]
    while process_running(proc)
      sleep(5)
    end
    close(ios[i])
    run(`cat $(paths[i])`)
  end
  @testset verbose=true "All" begin
    for (i,proc) ∈ enumerate(processes)
      @testset "part$i" begin
        @test success(proc)
      end
    end
  end
else
  include("grouptests.jl")
end






