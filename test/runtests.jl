include("testsetup.jl")

import InteractiveUtils

InteractiveUtils.versioninfo(stdout; verbose = true)

const LOOPVECTORIZATION_TEST = get(ENV, "LOOPVECTORIZATION_TEST", "all")

if LOOPVECTORIZATION_TEST == "all"
  NUMGROUPS = 6
  processes = Vector{Base.Process}(undef, NUMGROUPS)
  paths = Vector{String}(undef, NUMGROUPS)
  ios = Vector{IOStream}(undef, NUMGROUPS)
  tmp = tempdir()
  for i ∈ 1:NUMGROUPS
    path, io = mktemp(tmp)
    paths[i] = path
    ios[i] = io
    env = copy(ENV)
    env["LOOPVECTORIZATION_TEST"] = "part$i"
    env["JULIA_NUM_THREADS"] = string(Threads.nthreads())
    processes[i] = run(
      pipeline(
        setenv(`$(Base.julia_cmd()) $(@__FILE__) --project=$(Base.active_project())`, env),
        stderr = io,
        stdout = io,
      ),
      wait = false,
    )
  end
  completed = fill(false, NUMGROUPS)
  while true
    for i ∈ eachindex(completed)
      completed[i] && continue
      proc = processes[i]
      process_running(proc) && continue
      completed[i] = true
      close(ios[i])
      println("Test Group: $i")
      run(`cat $(paths[i])`)
      println("\n\n")
    end
    all(completed) && break
    sleep(5)
  end
  @testset verbose = true "All" begin
    for (i, proc) ∈ enumerate(processes)
      @testset "part$i" begin
        @test success(proc)
      end
    end
  end
else
  include("grouptests.jl")
end
