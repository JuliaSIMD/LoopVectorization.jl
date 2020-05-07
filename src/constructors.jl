
### This file contains convenience functions for constructing LoopSets.

function Base.copyto!(ls::LoopSet, q::Expr)
    q.head === :for || throw("Expression must be a for loop.")
    add_loop!(ls, q, 8)
end

function add_ci_call!(q::Expr, @nospecialize(f), args, syms, i, mod = nothing)
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
    mod === nothing || push!(call.args, Expr(:call, Expr(:curly, :Val, QuoteNode(mod))))
    push!(q.args, Expr(:(=), syms[i], call))
end

function substitute_broadcast(q::Expr, mod::Symbol)
    ci = first(Meta.lower(LoopVectorization, q).args).code
    nargs = length(ci)-1
    ex = Expr(:block,)
    syms = [gensym() for _ ∈ 1:nargs]
    for n ∈ 1:nargs
        ciₙ = ci[n]
        ciₙargs = ciₙ.args
        f = first(ciₙargs)
        if ciₙ.head === :(=)
            push!(ex.args, Expr(:(=), f, syms[((ciₙargs[2])::Core.SSAValue).id]))
        elseif isglobalref(f, Base, :materialize!)
            add_ci_call!(ex, lv(:vmaterialize!), ciₙargs, syms, n, mod)
        elseif isglobalref(f, Base, :materialize)
            add_ci_call!(ex, lv(:vmaterialize), ciₙargs, syms, n, mod)
        else
            add_ci_call!(ex, f, ciₙargs, syms, n)
        end
    end
    ex
end


function LoopSet(q::Expr, mod::Symbol = :Main)
    SIMDPirates.contract_pass!(q)
    ls = LoopSet(mod)
    copyto!(ls, q)
    resize!(ls.loop_order, num_loops(ls))
    ls
end
LoopSet(q::Expr, m::Module) = LoopSet(macroexpand(m, q), Symbol(m))

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

`unroll` is an integer that specifies the loop unrolling factor, or a
tuple `(u₁, u₂) = (4, 2)` signaling that the generated code should unroll more than
one loop. `u₁` is the unrolling factor for the first unrolled loop and `u₂` for the next (if present),
but it applies to the loop ordering and unrolling that will be chosen by LoopVectorization,
*not* the order in `body`.
`uᵢ=0` (the default) indicates that LoopVectorization should pick its own value,
and `uᵢ=-1` disables unrolling for the correspond loop.

The `@avx` macro also checks the array arguments using `LoopVectorization.check_args` to try and determine
if they are compatible with the macro. If `check_args` returns false, a fall back loop annotated with `@inbounds`
and `@fastmath` is generated. Note that `SIMDPirates` provides functions such as `evadd` and `evmul` that will
ignore `@fastmath`, preserving IEEE semantics both within `@avx` and `@fastmath`.
`check_args` currently returns false for some wrapper types like `LinearAlgebra.UpperTriangular`, requiring you to
use their `parent`. Triangular loops aren't yet supported.
"""
macro avx(q)
    q = macroexpand(__module__, q)
    q2 = if q.head === :for
        setup_call(LoopSet(q, __module__), q)
    else# assume broadcast
        substitute_broadcast(q, Symbol(__module__))
    end
    esc(q2)
end

function check_inline(arg)
    a1 = (arg.args[1])::Symbol
    a1 === :inline || return zero(Int8)
    i = (arg.args[2])::Bool % Int8
    i + i - one(Int8)
end
function check_unroll(arg)
    a1 = (arg.args[1])::Symbol
    default = (zero(Int8),zero(Int8))
    a1 === :unroll || return default
    tup = arg.args[2]
    u₂ = -one(Int8)
    if tup isa Integer
        u₁ = convert(Int8, tup)
    elseif isa(tup, Expr)
        if length(tup.args) == 1
            u₁ = convert(Int8, tup.args[1])
        elseif length(tup.args) == 2
            u₁ = convert(Int8, tup.args[1])
            u₂ = convert(Int8, tup.args[2])
        else
            return default
        end
    else
        return default
    end
    u₁, u₂
end
function check_macro_kwarg(arg, inline::Int8 = zero(Int8), u₁::Int8 = zero(Int8), u₂::Int8 = zero(Int8))
    @assert arg.head === :(=)
    i = check_inline(arg)
    if iszero(i)
        u₁, u₂ = check_unroll(arg)
    else
        inline = i
    end
    inline, u₁, u₂
end
macro avx(arg, q)
    @assert q.head === :for
    @assert arg.head === :(=)
    q = macroexpand(__module__, q)
    inline, u₁, u₂ = check_macro_kwarg(arg)
    ls = LoopSet(q, __module__)
    esc(setup_call(ls, q, inline, u₁, u₂))
end
macro avx(arg1, arg2, q)
    @assert q.head === :for
    q = macroexpand(__module__, q)
    inline, u₁, u₂ = check_macro_kwarg(arg1)
    inline, u₁, u₂ = check_macro_kwarg(arg2, inline, u₁, u₂)
    esc(setup_call(LoopSet(q, __module__), q, inline, u₁, u₂))
end


"""
    @_avx

This macro transforms loops similarly to [`@avx`](@ref).
While `@avx` punts to a generated function to enable type-based analysis, `_@avx`
works on just the expressions. This requires that it makes a number of default assumptions.
"""
macro _avx(q)
    q = macroexpand(__module__, q)
    esc(lower_and_split_loops(LoopSet(q, __module__), -1))
end
macro _avx(arg, q)
    @assert q.head === :for
    q = macroexpand(__module__, q)
    inline, u₁, u₂ = check_macro_kwarg(arg)
    esc(lower(LoopSet(q, __module__), u₁ % Int, u₂ % Int, -1))
end

macro avx_debug(q)
    q = macroexpand(__module__, q)
    esc(LoopVectorization.setup_call_debug(LoopSet(q, __module__)))
end
