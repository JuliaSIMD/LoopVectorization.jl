# LoopVectorization

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://chriselrod.github.io/LoopVectorization.jl/stable)
[![Latest](https://img.shields.io/badge/docs-latest-blue.svg)](https://chriselrod.github.io/LoopVectorization.jl/latest)
[![Build Status](https://travis-ci.com/chriselrod/LoopVectorization.jl.svg?branch=master)](https://travis-ci.com/chriselrod/LoopVectorization.jl)
[![Build Status](https://ci.appveyor.com/api/projects/status/github/chriselrod/LoopVectorization.jl?svg=true)](https://ci.appveyor.com/project/chriselrod/LoopVectorization-jl)
[![Codecov](https://codecov.io/gh/chriselrod/LoopVectorization.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/chriselrod/LoopVectorization.jl)

## Installation
```
using Pkg
Pkg.add(PackageSpec(url="https://github.com/chriselrod/VectorizationBase.jl"))
Pkg.add(PackageSpec(url="https://github.com/chriselrod/SIMDPirates.jl"))
Pkg.add(PackageSpec(url="https://github.com/chriselrod/SLEEFPirates.jl"))
Pkg.add(PackageSpec(url="https://github.com/chriselrod/LoopVectorization.jl"))
```


## Usage

The current version of LoopVectorization provides a simple, dumb, transform on a single loop.
What I mean by this is that it will not check for the transformations for validity. To be safe, I would straight loops that transform arrays or calculate reductions.

For example,
```julia
function sum_simd(x)
    s = zero(eltype(x))
    @simd for xᵢ ∈ x
        s += xᵢ
    end
    s
end
using LoopVectorization, BenchmarkTools
function sum_loopvec(x::AbstractVector{Float64})
    s = 0.0
    @vvectorize for i ∈ eachindex(x)
        s += x[i]
    end
    s
end
x = rand(99);
@btime sum($x)

@btime sum_simd($x)
@btime sum_loopvec($x)
```



