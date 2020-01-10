
using BenchmarkTools

suite = BenchmarkGroup()
suite["linalg"] = BenchmarkGroup(["matmul","dot"])

include(joinpath(@__DIR__, "looptests.jl"))

for n ∈ 1:256

end
