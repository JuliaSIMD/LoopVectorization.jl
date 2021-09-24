# Composite Types: DateTime Arrays

Currently, loops over *some* types are easier to work with than others.
Here we show: (1) a sequential loop over `Vector{DateTime}` that cannot have
`@turbo` applied directly, and (2) a solution that uses the interpreted
integer representation of `DateTime`.

This may be applicable if you have a composite type that may be represented
with primitive types.

## Setting up the Problem

Here's a simple problem involving timestamps:

**Problem statement**:

- *Given*: a vector of *strictly increasing* timestamps.
- *Output*: a vector of the same length starting at `0.0` and ending at `1.0`.
  Each intermediate element is scaled proportionally to the length of time since
  the beginning.

**Sample Output**:

```julia
using Dates

sample_input = [
    Dates.DateTime(2021, 5, 5, 10, 0, 0),
    Dates.DateTime(2021, 5, 5, 10, 5, 15),
    Dates.DateTime(2021, 5, 6, 10, 0, 0),
    Dates.DateTime(2021, 5, 6, 10, 5, 15),
    Dates.DateTime(2021, 5, 7, 10, 0, 20),
]

expected_output = [
    0.0,
    0.0018227057053581761,
    0.499942136326814,
    0.5017648420321722,
    1.0,
]
```

## First Attempt: Sequential version of the loop

This implementation satisfies the problem statement by iterating over the
examples:

```julia
using Dates

function scale_timeseries_sequential(data::Vector{Dates.DateTime})
  out = similar(data, Float64)
  ϕ = (data[lastindex(data)] - data[1]).value

  @inbounds for i ∈ eachindex(data)
      out[i] = (data[i] - data[1]).value / ϕ
  end

  return out
end
```

## Second Attempt: Turbo Loop

Our `Vector{Dates.DateTime}` has an integer interpretation which we can take
advantage of here. We'll `reinterpret` our vector as `Int`, make the needed
adjustments, then apply the `@turbo` macro to our loop:

```julia
using LoopVectorization, Dates

function scale_timeseries_turbo(data::Vector{Dates.DateTime})

  # Interpret our DateTime vector as Int
  tsi = reinterpret(Int, data)

  out = similar(data, Float64)

  # We've interpreted our data as integers, so we no longer need `.value`
  ϕ = tsi[lastindex(tsi)] - tsi[1]

  @turbo for i ∈ eachindex(tsi)
      out[i] = (tsi[i] - tsi[1]) / ϕ
  end

  return out
end
```

## Benchmarks

We'll benchmark with randomly generated data:

```julia
function generate_timestamps(N::Int64)
    data = Vector{Dates.DateTime}(undef,N)
    v = DateTime(1990, 1, 1, 0, 0, 0)
    for i in 1:N
        v += Second(rand(1:5, 1)[1])
        data[i] =v
    end
    return data
end
```

Briefly, the benchmark suggests that the mean time for the sequential vs.
turbo solution is a ~3x speedup while holding memory requirements constant:

```julia
julia> using BenchmarkTools

julia> data_100000 = generate_timestamps(100000);

julia> data_200000 = generate_timestamps(200000);

julia> @benchmark scale_timeseries_sequential(data_100000)
BenchmarkTools.Trial: 10000 samples with 1 evaluation.
 Range (min … max):  318.864 μs … 967.760 μs  ┊ GC (min … max): 0.00% … 40.41%
 Time  (median):     321.291 μs               ┊ GC (median):    0.00%
 Time  (mean ± σ):   332.503 μs ±  52.040 μs  ┊ GC (mean ± σ):  1.97% ±  6.98%

  █▆▅▂▂▂▁                                                       ▁
  █████████▆▆▆▅▅▅▅▅▄▄▄▄▁▁▄▁▁▃▁▁▄▄▃▃▁▃▄▁▃▁▁▁▃▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▅█▇ █
  319 μs        Histogram: log(frequency) by time        701 μs <

 Memory estimate: 781.33 KiB, allocs estimate: 2.

julia> @benchmark scale_timeseries_turbo(data_100000)
BenchmarkTools.Trial: 10000 samples with 1 evaluation.
 Range (min … max):   71.942 μs … 933.400 μs  ┊ GC (min … max):  0.00% … 71.93%
 Time  (median):      87.926 μs               ┊ GC (median):     0.00%
 Time  (mean ± σ):   100.082 μs ±  89.095 μs  ┊ GC (mean ± σ):  11.63% ± 11.43%

  ▄█▃▁                                                        ▁ ▁
  ████▇▄▁▁▁▁▁▁▁▁▃▄▆▄▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█ █
  71.9 μs       Histogram: log(frequency) by time        764 μs <

 Memory estimate: 781.33 KiB, allocs estimate: 2.

julia> @benchmark scale_timeseries_sequential(data_200000)
BenchmarkTools.Trial: 7153 samples with 1 evaluation.
 Range (min … max):  637.692 μs …   2.277 ms  ┊ GC (min … max): 0.00% … 65.01%
 Time  (median):     640.729 μs               ┊ GC (median):    0.00%
 Time  (mean ± σ):   694.282 μs ± 184.965 μs  ┊ GC (mean ± σ):  3.69% ±  8.68%

  █▆▅▃▂▁                                                        ▁
  ███████▇▅▆▆▄▄▅▁▁▁▁▁▁▆██▇▅▄▄▁▃▁▃▃▄▁▁▁▁▁▁▁▁▁▁▁▇█▇▇▅▅▄▃▄▄▁▁▁▄▄█▇ █
  638 μs        Histogram: log(frequency) by time       1.71 ms <

 Memory estimate: 1.53 MiB, allocs estimate: 2.

julia> @benchmark scale_timeseries_turbo(data_200000)
BenchmarkTools.Trial: 10000 samples with 1 evaluation.
 Range (min … max):  159.023 μs …   2.092 ms  ┊ GC (min … max):  0.00% … 50.30%
 Time  (median):     176.559 μs               ┊ GC (median):     0.00%
 Time  (mean ± σ):   230.513 μs ± 189.542 μs  ┊ GC (mean ± σ):  11.86% ± 12.80%

  █▇▅▄▄▃▂▂▁          ▁▁                                         ▂
  ██████████▇▅▅▃▁▄▁▃▁██▇▆▄▅▄▄▅▄▅▃▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁███▇▅▅▅▅▄▅▄▅█▇▇ █
  159 μs        Histogram: log(frequency) by time       1.22 ms <

 Memory estimate: 1.53 MiB, allocs estimate: 2.
```
