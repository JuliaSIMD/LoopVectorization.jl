
### This file contains convenience functions for constructing LoopSets.

# function strip_unneeded_const_deps!(ls::LoopSet)
#     for op ∈ operations(ls)
#         if isconstant(op) && iszero(length(reducedchildren(op)))
#             op.dependencies = NODEPENDENCY
#         end
#     end
#     ls
# end

function Base.copyto!(ls::LoopSet, q::Expr)
    q.head === :for || throw("Expression must be a for loop.")
    add_loop!(ls, q, 8)
    # strip_unneeded_const_deps!(ls)
end

function add_ci_call!(q::Expr, @nospecialize(f), args, syms, i, valarg = nothing, mod = nothing)
    call = if f isa Core.SSAValue
        Expr(:call, syms[f.id])
    else
        Expr(:call, f)
    end
    for arg ∈ @view(args[2:end])
        if arg isa Core.SSAValue
            push!(call.args, syms[arg.id])
        else
            push!(call.args, arg)
        end
    end
    if mod !== nothing # indicates it's `vmaterialize(!)`
        reg_size = Expr(:call, lv(:register_size))
        reg_count = Expr(:call, lv(:available_registers))
        cache_lnsze = Expr(:call, lv(:cache_linesize))
        push!(call.args, Expr(:call, Expr(:curly, :Val, QuoteNode(mod))), valarg, reg_size, reg_count, cache_lnsze)
    end
    push!(q.args, Expr(:(=), syms[i], call))
end

function substitute_broadcast(q::Expr, mod::Symbol, inline, u₁, u₂, threads)
    ci = first(Meta.lower(LoopVectorization, q).args).code
    nargs = length(ci)-1
    ex = Expr(:block,)
    syms = [gensym() for _ ∈ 1:nargs]
    valarg = :(Val{$(inline, u₁, u₂, threads)}())
    for n ∈ 1:nargs
        ciₙ = ci[n]
        ciₙargs = ciₙ.args
        f = first(ciₙargs)
        if ciₙ.head === :(=)
            push!(ex.args, Expr(:(=), f, syms[((ciₙargs[2])::Core.SSAValue).id]))
        elseif isglobalref(f, Base, :materialize!)
            add_ci_call!(ex, lv(:vmaterialize!), ciₙargs, syms, n, valarg, mod)
        elseif isglobalref(f, Base, :materialize)
            add_ci_call!(ex, lv(:vmaterialize), ciₙargs, syms, n, valarg, mod)
        else
            add_ci_call!(ex, f, ciₙargs, syms, n)
        end
    end
    ex
end


function LoopSet(q::Expr, mod::Symbol = :Main)
    contract_pass!(q)
    ls = LoopSet(mod)
    copyto!(ls, q)
    resize!(ls.loop_order, num_loops(ls))
    ls
end
LoopSet(q::Expr, m::Module) = LoopSet(macroexpand(m, q)::Expr, Symbol(m))

function loopset(q::Expr) # for interactive use only
    ls = LoopSet(q)
    set_hw!(ls)
    ls
end

function check_macro_kwarg(arg, inline::Bool, check_empty::Bool, u₁::Int8, u₂::Int8, threads::Int)
    ((arg.head === :(=)) && (length(arg.args) == 2)) || throw(ArgumentError("macro kwarg should be of the form `argname = value`."))
    kw = (arg.args[1])::Symbol
    value = (arg.args[2])
    if kw === :inline
        inline = value::Bool
    elseif kw === :unroll
        if value isa Integer
            u₁ = convert(Int8, tup)::Int8
        elseif Meta.isexpr(value,:tuple,2)
            u₁ = convert(Int8, tup.args[1])::Int8
            u₂ = convert(Int8, tup.args[2])::Int8
        else
            throw(ArgumentError("Don't know how to process argument in `unroll=$value`."))
        end
    elseif kw === :check_empty
        check_empty = value::Bool
    elseif kw === :thread
        if value isa Bool
            threads = Core.ifelse(value::Bool, -1, 1)
        elseif value isa Integer
            threads = max(1, convert(Int,value)::Int)
        else
            throw(ArgumentError("Don't know how to process argument in `thread=$value`."))
        end
    else
        throw(ArgumentError("Received unrecognized keyword argument $kw. Recognized arguments include:\n`inline`, `unroll`, `check_empty`, and `thread`."))
    end
    inline, check_empty, u₁, u₂, threads
end
function avx_macro(mod, src, q, args...)
    q = macroexpand(mod, q)
    inline = false; check_empty = false; u₁ = zero(Int8); u₂ = zero(Int8); threads = 1;
    for arg ∈ args
        inline, check_empty, u₁, u₂, threads = check_macro_kwarg(arg, inline, check_empty, u₁, u₂, threads)
    end
    ls = LoopSet(q, mod)
    if q.head === :for
        esc(setup_call(ls, q, src, inline, check_empty, u₁, u₂, threads))
    else
        substitute_broadcast(q, Symbol(mod), inline, u₁, u₂, threads)
    end
end
"""
    @avx

Annotate a `for` loop, or a set of nested `for` loops whose bounds are constant across iterations, to optimize the computation. For example:

    function AmulBavx!(C, A, B)
        @avx for m ∈ 1:size(A,1), n ∈ 1:size(B,2)
            Cₘₙ = zero(eltype(C))
            for k ∈ 1:size(A,2)
                Cₘₙ += A[m,k] * B[k,n]
            end
            C[m,n] = Cₘₙ
        end
    end

The macro models the set of nested loops, and chooses an ordering of the three loops to
minimize predicted computation time.

It may also apply to broadcasts:

```jldoctest
julia> using LoopVectorization

julia> a = rand(100);

julia> b = @avx exp.(2 .* a);

julia> c = similar(b);

julia> @avx @. c = exp(2a);

julia> b ≈ c
true
```

# Extended help

Advanced users can customize the implementation of the `@avx`-annotated block
using keyword arguments:

```
@avx inline=false unroll=2 body
```

where `body` is the code of the block (e.g., `for ... end`).

`thread` is either a Boolean, or an integer.
The integer's value indicates the number of threads to use.
It is clamped to be between `1` and `min(Threads.nthreads(),LoopVectorization.num_cores())`.
`false` is equivalent to `1`, and `true` is equivalent to `min(Threads.nthreads(),LoopVectorization.num_cores())`.

`inline` is a Boolean. When `true`, `body` will be directly inlined
into the function (via a forced-inlining call to `_avx_!`).
When `false`, it wont force inlining of the call to `_avx_!` instead, letting Julia's own inlining engine
determine whether the call to `_avx_!` should be inlined. (Typically, it won't.)
Sometimes not inlining can lead to substantially worse code generation, and >40% regressions, even in very
large problems (2-d convolutions are a case where this has been observed).
One can find some circumstances where `inline=true` is faster, and other circumstances
where `inline=false` is faster, so the best setting may require experimentation. By default, the macro
tries to guess. Currently the algorithm is simple: roughly, if there are more than two dynamically sized loops
or and no convolutions, it will probably not force inlining. Otherwise, it probably will.

`check_empty` (default is `false`) determines whether or not it will check if any of the iterators are empty.
If false, you must ensure yourself that they are not empty, else the behavior of the loop is undefined and
(like with `@inbounds`) segmentation faults are likely.

`unroll` is an integer that specifies the loop unrolling factor, or a
tuple `(u₁, u₂) = (4, 2)` signaling that the generated code should unroll more than
one loop. `u₁` is the unrolling factor for the first unrolled loop and `u₂` for the next (if present),
but it applies to the loop ordering and unrolling that will be chosen by LoopVectorization,
*not* the order in `body`.
`uᵢ=0` (the default) indicates that LoopVectorization should pick its own value,
and `uᵢ=-1` disables unrolling for the correspond loop.

The `@avx` macro also checks the array arguments using `LoopVectorization.check_args` to try and determine
if they are compatible with the macro. If `check_args` returns false, a fall back loop annotated with `@inbounds`
and `@fastmath` is generated. Note that `VectorizationBase` provides functions such as `vadd` and `vmul` that will
ignore `@fastmath`, preserving IEEE semantics both within `@avx` and `@fastmath`.
`check_args` currently returns false for some wrapper types like `LinearAlgebra.UpperTriangular`, requiring you to
use their `parent`. Triangular loops aren't yet supported.
"""
macro avx(args...)
    avx_macro(__module__, __source__, last(args), Base.front(args)...)
end
"""
Equivalent to `@avx`, except it adds `thread=true` as the first keyword argument.
Note that later arguments take precendence.

Meant for convenience, as `@avxt` is shorter than `@avx thread=true`.
"""
macro avxt(args...)
    avx_macro(__module__, __source__, last(args), :(thread=true), Base.front(args)...)
end

"""
    @_avx

This macro transforms loops similarly to [`@avx`](@ref).
While `@avx` punts to a generated function to enable type-based analysis, `_@avx`
works on just the expressions. This requires that it makes a number of default assumptions. Use of `@avx` is preferred.

This macro accepts the `inline` and `unroll` keyword arguments like `@avx`, but ignores the `check_empty` argument.
"""
macro _avx(q)
    q = macroexpand(__module__, q)
    ls = LoopSet(q, __module__)
    set_hw!(ls)
    esc(Expr(:block, ls.prepreamble, lower_and_split_loops(ls, -1)))
end
macro _avx(arg, q)
    @assert q.head === :for
    q = macroexpand(__module__, q)
    inline, check_empty, u₁, u₂ = check_macro_kwarg(arg)
    ls = LoopSet(q, __module__)
    set_hw!(ls)
    esc(Expr(:block, ls.prepreamble, lower(ls, u₁ % Int, u₂ % Int, -1)))
end

macro avx_debug(q)
    q = macroexpand(__module__, q)
    ls = LoopSet(q, __module__)
    esc(LoopVectorization.setup_call_debug(ls))
end
