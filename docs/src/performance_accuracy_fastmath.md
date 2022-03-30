## Performance vs. accuracy and comparison with `@fastmath`

_From the Julia slack #performance-helpdesk 2022-03-30 03:06 UTC_

**Daniel Wennberg**

Does `@turbo` from LoopVectorization, and by extension `@tullio`, have the same accuracy caveats as `@fastmath`, as discussed e.g. here: <https://discourse.julialang.org/t/fastmath-macro-accuracy/38847/7>? Seeing as LoopVectorization's fallback mode is `@inbounds` `@fastmath`

**Chris Elrod**

More often than not, the transforms enabled by `@turbo` and `@fastmath` are going to make results more accurate.

I.e., allowing use of multiple accumulators and fma instructions will help. (edited) 

**Chris Elrod**

`@fastmath` applied globally makes results less accurate on average, because there are some extreme cases, where code was written with IEEE semantics in mind, performing compensated arithmetic/error accumulation and adjustment.

By allowing reassociating this code, errors can become catastrophic.

If you did not write code in such a deliberate manner, you won’t encounter the problem when applying it to your own code.

It is unsafe to apply to other people’s code, unless you happen to know they also did not write the specific sequence of floating point operations in such an intentional way.

Many functions, like `exp` and `sinpi` are written in such an intentional way, and thus experience terrible error when starting julia with --math-mode=fast. But `@fastmath exp(x)` switches `exp` implementations to a less accurate, slightly faster, version, rather than applying `@fastmath` to the contents of the regular exp, so it is not dangerous to do.

I think the fear mongering over fastmath comes from languages like C/C++/Fortran, where you apply it either everywhere or nowhere.

Applying it everywhere is dangerous, because then it gets more and more likely that some code or some library was written in such a way with error tracking, that it causes catastrophic problems.

Applying it locally is the correct approach.

As would be, IMO, allowing at least associative math everywhere, and letting people opt out in the cases where they’re deliberately avoiding it.

**Chris Elrod**

`@fastmath` and `@turbo` also enable less accurate functions, like less accurate implementations of `^`, `sin`, etc.

**Chris Elrod**

But especially in the case of `@turbo`, these versions are way faster. Most of these should still be good to 3 ULP (units in last place)

**Oscar Smith**

does `@turbo` have the no nan parts of fastmath?

**Daniel Wennberg**

Thanks, this is super helpful!

**Daniel Wennberg**

> But especially in the case of `@turbo`, these versions are way faster.

So `@turbo` and `@fastmath` substitute different fast versions of elementary functions? Is this where `@turbo` makes better use of the AVX instruction set?

**Oscar Smith**

yes

**Oscar Smith**

regular `@fastmath` uses slightly faster scalar functions which LLVM may autovectorize, but `@fastmath` isn't a powerful enough macro to tell when when a function is called in a loop

**Chris Elrod**

No, `@turbo` does not have the no nans part.

**Chris Elrod**

Unfortunately, no nans propagates in a way that is at high risk of disabling checks for nans, even when only applying it locally, making no nans basically unusable IMO.

**Chris Elrod**

`@turbo`’s speed gain is mostly from the fact it has versions of special functions that are SIMD-able, not from the lower accuracy.

**Chris Elrod**  2 hours ago

You should be able to manually opt into accurate, but still SIMD-able, versions.

**Alec**

@Chris Elrod you should crosspost that to discourse for posterity's sake

**Daniel Wennberg**

> You should be able to manually opt into accurate, but still SIMD-able, versions.

You mean a user like me can do that currently or are you saying this as a feature request to yourself/Julia/LLVM?

**Daniel Wennberg**

Is this what `VectorizationBase.vadd` etc. enable?

**Oscar Smith**

it's a feature request. The `VectorizationBase.v` versions are the fast but slightly inaccurate ones

**Chris Elrod**

```julia
julia> @inline sincos_simd(x) = SLEEFPirates.sincos(x)
sincos_simd (generic function with 1 method)

julia> function sincos_turbo_accurate!(sc, x)
           @turbo for i ∈ eachindex(x)
               sc[i,1], sc[i,2] = sincos_simd(x[i])
           end
       end
sincos_turbo_accurate! (generic function with 1 method)

julia> function sincos_turbo!(sc, x)
           @turbo for i ∈ eachindex(x)
               sc[i,1], sc[i,2] = sincos(x[i])
           end
       end
sincos_turbo! (generic function with 1 method)

julia> function sincos!(sc, x)
           @inbounds for i ∈ eachindex(x)
               sc[i,1], sc[i,2] = sincos(x[i])
           end
       end
sincos! (generic function with 1 method)

julia> function sincos_fast!(sc, x)
           @inbounds @fastmath for i ∈ eachindex(x)
               sc[i,1], sc[i,2] = sincos(x[i])
           end
       end
sincos_fast! (generic function with 1 method)

julia> x = rand(512); sc = similar(x, length(x), 2);

julia> @btime sincos!($sc,$x)
  3.918 μs (0 allocations: 0 bytes)

julia> @btime sincos_fast!($sc,$x)
  3.906 μs (0 allocations: 0 bytes)

julia> @btime sincos_turbo!($sc,$x)
  614.067 ns (0 allocations: 0 bytes)

julia> @btime sincos_turbo_accurate!($sc,$x)
  1.702 μs (0 allocations: 0 bytes)
```

**Chris Elrod**

The rewrite will probably respect regular fastmath, or have a flag for that.

**Chris Elrod**

Anyway, if you want to use the more accurate versions of functions like `sincos`, you can hide `SLEEFPirates.sincos` from LV behind a function, like in the above example

**Oscar Smith**

Also, have you seen <https://github.com/ARM-software/optimized-routines/tree/master/math>? It's a really good set of elementary implementations that should vectorize pretty well. Their `pow` is of special interest (and their `log` and `exp` are really good too) I'm in the processes of rewriting the `Base` versions to use some of their tricks

**Oscar Smith**

the one downside of them from a vectorizaiton standpoint is they do slightly more math in `UInt` form, but most of that is unnecessary

**Chris Elrod**

It’ll be a while until I get that far on the rewrite!

The ideal would be that I can take your Julia implementations and analyze the LLVM to figure out how to vectorize them, without needing any separate implementations.

That is plan A.

**Chris Elrod**

What’s wrong with `UInt`, that AVX512DQ is needed on x86?

**Chris Elrod**

I’ve at least been seeing more rumors of AVX512 support in Zen4.

**Oscar Smith**

yeah, but Intel is mostly killing it. Also AVX2 is really common...

**Chris Elrod**

What operations, and can they be implemented by splitting into UInt32s? Not ideal, but still a net win if it enables SIMD…

**Chris Elrod**

Intel’s server CPUs still have AVX512

**Chris Elrod**

I’m still optimistic that it’ll return to consumer CPUs. I suspect it was intended for Alder Lake, but some sort of support to make it work with the efficiency cores wasn’t handled yet.

**Chris Elrod**

Maybe the successor to gracemont will support AVX512, in the same way Zen1 supported AVX2.

**Chris Elrod**

`UInt32`s don’t really work well for things like shifts, nor is there any real overflow detection support.

**Chris Elrod**

So I might need some sort of plan B, where I have to substitute `^` and friends for `@llvm.pow` or `@turbo.pow`, and then have my own implementations for these.

**Chris Elrod**

At least for a few critical/special interest functions.

**Daniel Wennberg**

Folks, the effort you're putting into helping people get the most out of their hardware with just some quick annotations and no detailed knowledge of simd and avx and special function implementations is hugely appreciated and I want you to give yourselves a pat on the back

**Chris Elrod**

@alecloudenback The thread is a couple years old, so I'm not eager to revive it. Not sure what the best way to save it for posterity is.

**Daniel Wennberg**

However, @Chris Elrod, your example left me wondering about exactly how `@turbo` (and `@fastmath` for that matter) interacts with function barriers and inlining. For example, how should I think about the behavior and performance of the three versions of f! in the following example?

```julia
const s = 1.2
f(x) = muladd(s, tanh(x), -x)
@inline f_inlined(x) = muladd(s, tanh(x), -x)

function f!(y, x)
    @turbo for i in eachindex(x)
        y[i] = f(x[i])
    end
end

function f_inlined!(y, x)
    @turbo for i in eachindex(x)
        y[i] = f_inlined(x[i])
    end
end

function f_inlined_manual!(y, x)
    @turbo for i in eachindex(x)
        y[i] = muladd(s, tanh(x[i]), -x[i])
    end
end

julia> x = randn(512); y = similar(x);

julia> @btime f!(y, x)
  3.153 μs (0 allocations: 0 bytes)

julia> @btime f_inlined!(y, x)
  2.966 μs (0 allocations: 0 bytes)

julia> @btime f_inlined_manual!(y, x)
  2.696 μs (0 allocations: 0 bytes)
```

**Chris Elrod**

`@fastmath` only works on syntax, replacing functions like `+` with `Base.FastMath.add_fast`, so hiding something behind a function will protect it from `@fastmath`.

**Daniel Wennberg**

> Not sure what the best way to save it for posterity is.

I generally appreciate when documentation includes discussions along these lines. In this case that would include both the LoopVectorization docs for everything that pertains to `@turbo`, as well as the base Julia docs for details about `@fastmath`, especially the difference between `@fastmath exp(x)` and `exp(x)` under `--math-mode=fast`---I think I was simply assuming that `@fastmath` was all about compiler flags and would transform the elementary arithmetic within each special function.

**Chris Elrod**

`@turbo` is similar, which is why the sincos example worked, with a few extra caveats:

1. it can still SIMD things behind a function, because it also uses the type system for vectorization.

2. `@turbo`'s understanding of what code does is based on syntax (with a little help from types), so if you hide a function from it, it will not know what that function does. It currently assumes the function is very expensive, and will not do anything fancy. In the case of `tanh`, that's close enough so you don't see much of a difference.

**Chris Elrod**

Actually, the difference there is probably because there is a separate `tanh` and `tanh_fast`.

Inlining can help a bit, because these functions generally involve polynomials with lots of constants. Inlining lets you avoid reloading these constants on every iteration by hoisting the loads out of the loop.

**Chris Elrod**

You're welcome to make a PR to add documentation based on this discussion.

**Oscar Smith**

I think `--math-mode=fast` should be disabled. It's never actually what you want. <https://github.com/JuliaLang/julia/pull/41638> was going to do it, but we got distracted and forgot.

**Daniel Wennberg**

My general takeaway from this discussion is that both `@fastmath` and `@turbo` are mostly safe to use on your own code unless a) you're knowingly exploiting IEEE semantics in accumulations (in which case you wouldn't dream of using these macros anyway), or b) you depend on the 1 ULP accuracy of the standard implementation of a special function (do you though?). Does that sound about right? What I'm still not quite clear on is the issue of the no nans flag in `@fastmath` being "basically unusable"---is it dangerous or just not helpful?

**Chris Elrod**

```julia
julia> function foo(a,b,c)
           d = @fastmath a + b
           e = @fastmath b * c
           f = d + e
           isnan(f)
       end
foo (generic function with 2 methods)

julia> @code_llvm debuginfo=:none foo(NaN,NaN,NaN)
define i8 @julia_foo_1015(double %0, double %1, double %2) #0 {
top:
  ret i8 0
}
```

Use of `@fastmath` enables the compiler to prove that down stream results also are not `NaN`.

I do not like to lose the ability to check for `NaN`s.

Yet the above function was compiled away to return `false`, i.e. that it is impossible for `f` to be `NaN`.

I wouldn't mind if `@fastmath` allowed optimizing just those functions as though they weren't `NaN`. If the compiler could prove `c` or `b` are `0.0`, this would allow eliminating the multiplication and then writing `f = d`.

But using that some operations are marked nonans to prove that others aren't `NaN` is going too far IMO.

**Chris Elrod**

That is, I do like being able to optimize particular code in a way that would change the answer in the presence of `NaN`s, but I do not actually want to promise the compiler that the values are not `NaN`s, as I'd still like to check for this later.

**Chris Elrod**

Checking arguments still works:

```julia
julia> function foo(a,b,c)
           d = @fastmath a + b
           e = @fastmath b * c
           f = d + e
           isnan(f) | isnan(b)
       end
foo (generic function with 2 methods)

julia> @code_llvm debuginfo=:none foo(NaN,NaN,NaN)
define i8 @julia_foo_1019(double %0, double %1, double %2) #0 {
top:
  %3 = fcmp uno double %1, 0.000000e+00
  %4 = zext i1 %3 to i8
  ret i8 %4
} 
```

But sometimes it is easier to check results, e.g. if the arguments are hidden inside some other function.

**Daniel Wennberg**

Looks dangerous to me. So another heuristic for when to avoid `@fastmath`: c) downstream behavior depends on whether the result was inf or nan (there's a similar flag for inf that's also enabled, right?)

**Chris Elrod**

Yes, there is a similar flag for `Inf`.

**Chris Elrod**

LoopVectorization does not apply that either.

**Daniel Wennberg**

Just one more clarification question if you can be bothered: returning to the function barrier/inlining example, do I understand correctly that:

* `f!` can SIMD
* `f_inlined!` can SIMD and hoist constants within `tanh` out of the loop
* `f_inlined_manual!` can SIMD, replace `tanh` with `tanh_fast`, and hoist constants within `tanh_fast` out of the loop

**Chris Elrod**

Yes
