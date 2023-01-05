
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

function add_ci_call!(
  q::Expr,
  @nospecialize(f),
  args::Vector{Any},
  syms::Vector{Symbol},
  i::Int,
  @nospecialize(valarg) = nothing,
  @nospecialize(mod) = nothing,
)
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
    push!(call.args, Expr(:call, Expr(:curly, :Val, QuoteNode(mod))), valarg)
  end
  push!(q.args, Expr(:(=), syms[i], call))
end

function substitute_broadcast(
  q::Expr,
  mod::Symbol,
  inline::Bool,
  u₁::Int8,
  u₂::Int8,
  v::Int8,
  threads::Int,
  warncheckarg::Int,
  safe::Bool,
)
  ci = first(Meta.lower(LoopVectorization, q).args).code
  nargs = length(ci) - 1
  lb = Expr(:block)
  syms = Vector{Symbol}(undef, nargs)
  configarg = (inline, u₁, u₂, v, true, threads, warncheckarg, safe)
  unroll_param_tup = Expr(:call, lv(:avx_config_val), :(Val{$configarg}()), staticexpr(0))
  for n ∈ 1:nargs
    _ciₙ = ci[n]
    if _ciₙ isa Symbol
      syms[n] = _ciₙ::Symbol
    else
      syms[n] = Symbol('%', n)
      ciₙ::Expr = _ciₙ::Expr
      ciₙargs = ciₙ.args
      f = first(ciₙargs)
      if ciₙ.head === :(=)
        push!(lb.args, Expr(:(=), f, syms[((ciₙargs[2])::Core.SSAValue).id]))
      elseif isglobalref(f, Base, :materialize!)
        add_ci_call!(lb, lv(:vmaterialize!), ciₙargs, syms, n, unroll_param_tup, mod)
      elseif isglobalref(f, Base, :materialize)
        add_ci_call!(lb, lv(:vmaterialize), ciₙargs, syms, n, unroll_param_tup, mod)
      else
        add_ci_call!(lb, f, ciₙargs, syms, n)
      end
    end
  end
  ret::Expr = pop!(lb.args)::Expr
  if Meta.isexpr(ret, :(=), 2)
    ret = (ret.args[2])::Expr
  end
  esc(Expr(:let, lb, Expr(:block, ret)))
end


function LoopSet(q::Expr, mod::Symbol = :Main)
  ls = LoopSet(mod)
  check_inputs!(q, ls.prepreamble)
  contract_pass!(q)
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

function check_macro_kwarg(
  arg,
  inline::Bool,
  check_empty::Bool,
  u₁::Int8,
  u₂::Int8,
  v::Int8,
  threads::Int,
  warncheckarg::Int,
  safe::Bool,
)
  ((arg.head === :(=)) && (length(arg.args) == 2)) ||
    throw(ArgumentError("macro kwarg should be of the form `argname = value`."))
  kw = (arg.args[1])::Symbol
  value = (arg.args[2])
  if kw === :inline
    inline = value::Bool
  elseif kw === :unroll
    if value isa Integer
      u₁ = convert(Int8, value)::Int8
    elseif Meta.isexpr(value, :tuple, 2)
      u₁ = convert(Int8, value.args[1])::Int8
      u₂ = convert(Int8, value.args[2])::Int8
    else
      throw(ArgumentError("Don't know how to process argument in `unroll=$value`."))
    end
  elseif kw === :vectorize
    v = convert(Int8, value)
  elseif kw === :check_empty
    check_empty = value::Bool
  elseif kw === :thread
    if value isa Bool
      threads = Core.ifelse(value::Bool, -1, 1)
    elseif value isa Integer
      threads = max(1, convert(Int, value)::Int)
    else
      throw(ArgumentError("Don't know how to process argument in `thread=$value`."))
    end
  elseif kw === :warn_check_args
    warncheckarg = convert(Int, value)::Int
  elseif kw === :safe
    safe = convert(Bool, value)
  else
    throw(
      ArgumentError(
        "Received unrecognized keyword argument $kw. Recognized arguments include:\n`inline`, `unroll`, `check_empty`, and `thread`.",
      ),
    )
  end
  inline, check_empty, u₁, u₂, v, threads, warncheckarg, safe
end
function process_args(
  args;
  inline::Bool = false,
  check_empty::Bool = false,
  u₁::Int8 = zero(Int8),
  u₂::Int8 = zero(Int8),
  v::Int8 = zero(Int8),
  threads::Int = 1,
  warncheckarg::Int = 1,
  safe::Bool = true,
)
  for arg ∈ args
    inline, check_empty, u₁, u₂, v, threads, warncheckarg, safe =
      check_macro_kwarg(arg, inline, check_empty, u₁, u₂, v, threads, warncheckarg, safe)
  end
  inline, check_empty, u₁, u₂, v, threads, warncheckarg, safe
end
# check if the body of loop is a block, if not convert it to a block issue#395
# and check if the range of loop is an enumerate, if it is replace it, issue#393
function check_inputs!(q, prepreamble)
  if Meta.isexpr(q, :for)
    if !Meta.isexpr(q.args[2], :block)
      q.args[2] = Expr(:block, q.args[2])
      replace_enumerate!(q, prepreamble) # must after warp block
    else # maybe inner loops in block
      replace_enumerate!(q, prepreamble)
      for arg in q.args[2].args
        check_inputs!(arg, prepreamble) # check recursively for inner loop
      end
    end
  end
  return q
end
function replace_enumerate!(q, prepreamble)
  looprange = q.args[1]
  if Meta.isexpr(looprange, :block)
    for i = 1:length(looprange.args)
      replace_single_enumerate!(q, prepreamble, i)
    end
  else
    replace_single_enumerate!(q, prepreamble)
  end
  return q
end
function replace_single_enumerate!(q, prepreamble, i = nothing)
  if isnothing(i) # not nest loop
    looprange, body = q.args[1]::Expr, q.args[2]
  else # nest loop
    looprange, body = q.args[1].args[i]::Expr, q.args[2]
  end
  @assert Meta.isexpr(looprange, :(=), 2)
  itersyms, r = looprange.args
  if Meta.isexpr(r, :call, 2) && r.args[1] === :enumerate
    _iter = r.args[2]
    if _iter isa Symbol
      iter = _iter
    else # name complex expr
      iter = gensym(:iter)
      push!(prepreamble.args, :($iter = $_iter))
    end
    if Meta.isexpr(itersyms, :tuple, 2)
      indsym, varsym = itersyms.args[1]::Symbol, itersyms.args[2]::Symbol
      _replace_looprange!(q, i, indsym, iter)
      pushfirst!(body.args, :($varsym = $iter[$indsym+firstindex($iter)-1]))
    elseif Meta.isexpr(itersyms, :tuple, 1) # like `for (i,) in enumerate(...)`
      indsym = itersyms.args[1]::Symbol
      _replace_looprange!(q, i, indsym, iter)
    elseif itersyms isa Symbol # if itersyms are not unbox in loop range
      throw(ArgumentError("`for $itersyms in enumerate($r)` is not supported,
        please use `for ($(itersyms)_i, $(itersyms)_v) in enumerate($r)` instead."))
    else
      throw(ArgumentError("Don't know how to handle expression `$itersyms`."))
    end
  end
  return q
end
_replace_looprange!(q, ::Nothing, indsym, iter) =
  q.args[1] = :($indsym = Base.OneTo(length($iter)))
_replace_looprange!(q, i::Int, indsym, iter) =
  q.args[1].args[i] = :($indsym = Base.OneTo(length($iter)))

function turbo_macro(mod, src, q, args...)
  q = macroexpand(mod, q)
  if q.head === :for
    ls = LoopSet(q, mod)
    inline, check_empty, u₁, u₂, v, threads, warncheckarg, safe = process_args(args)
    esc(setup_call(ls, q, src, inline, check_empty, u₁, u₂, v, threads, warncheckarg, safe))
  else
    inline, check_empty, u₁, u₂, v, threads, warncheckarg, safe =
      process_args(args, inline = true)
    substitute_broadcast(q, Symbol(mod), inline, u₁, u₂, v, threads, warncheckarg, safe)
  end
end
"""
    @turbo

Annotate a `for` loop, or a set of nested `for` loops whose bounds are constant across iterations, to optimize the computation. For example:

    function AmulB!(C, A, B)
        @turbo for m ∈ indices((A,C), 1), n ∈ indices((B,C), 2) # indices((A,C),1) == axes(A,1) == axes(C,1)
            Cₘₙ = zero(eltype(C))
            for k ∈ indices((A,B), (2,1)) # indices((A,B), (2,1)) == axes(A,2) == axes(B,1)
                Cₘₙ += A[m,k] * B[k,n]
            end
            C[m,n] = Cₘₙ
        end
    end

The macro models the set of nested loops, and chooses an ordering of the three loops to minimize predicted computation time.

Current limitations:

1. It assumes that loop iterations are independent.
2. It does not perform bounds checks.
3. It assumes that each loop iterates at least once. (Use `@turbo check_empty=true` to lift this assumption.)
4. That there is only one loop at each level of the nest.

It may also apply to broadcasts:

```jldoctest
julia> using LoopVectorization

julia> a = rand(100);

julia> b = @turbo exp.(2 .* a);

julia> c = similar(b);

julia> @turbo @. c = exp(2a);

julia> b ≈ c
true
```

# Extended help

Advanced users can customize the implementation of the `@turbo`-annotated block
using keyword arguments:

```julia
@turbo inline=false unroll=2 thread=4 body
```

where `body` is the code of the block (e.g., `for ... end`).

`thread` is either a Boolean, or an integer.
The integer's value indicates the number of threads to use.
It is clamped to be between `1` and `min(Threads.nthreads(),LoopVectorization.num_cores())`.
`false` is equivalent to `1`, and `true` is equivalent to `min(Threads.nthreads(),LoopVectorization.num_cores())`.

`safe` (defaults to `true`) will cause `@turbo` to fall back to `@inbounds @fastmath` if `can_turbo` returns false for any of the functions called in the loop. You can disable the associated warning with `warn_check_args=false`.

Setting the keyword argument `warn_check_args=true` (e.g. `@turbo warn_check_args=true for ...`) in a loop or
broadcast statement will cause it to warn once if `LoopVectorization.check_args` fails and the fallback
loop is executed instead of the LoopVectorization-optimized loop.
Setting it to an integer > 0 will warn that many times, while setting it to a negative integer will warn
an unlimited amount of times. The default is `warn_check_args = 1`. Failure means that there may have been an array with unsupported type, unsupported element types, or (if `safe=true`) a function for which `can_turbo` returned `false`.

`inline` is a Boolean. When `true`, `body` will be directly inlined
into the function (via a forced-inlining call to `_turbo_!`).
When `false`, it wont force inlining of the call to `_turbo_!` instead, letting Julia's own inlining engine
determine whether the call to `_turbo_!` should be inlined. (Typically, it won't.)
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

The `@turbo` macro also checks the array arguments using `LoopVectorization.check_args` to try and determine
if they are compatible with the macro. If `check_args` returns false, a fall back loop annotated with `@inbounds`
and `@fastmath` is generated. Note that `VectorizationBase` provides functions such as `vadd` and `vmul` that will
ignore `@fastmath`, preserving IEEE semantics both within `@turbo` and `@fastmath`.
`check_args` currently returns false for some wrapper types like `LinearAlgebra.UpperTriangular`, requiring you to
use their `parent`. Triangular loops aren't yet supported.
"""
macro turbo(args...)
  turbo_macro(__module__, __source__, last(args), Base.front(args)...)
end
"""
    @tturbo

Equivalent to `@turbo`, except it adds `thread=true` as the first keyword argument.
Note that later arguments take precendence.

Meant for convenience, as `@tturbo` is shorter than `@turbo thread=true`.
"""
macro tturbo(args...)
  turbo_macro(__module__, __source__, last(args), :(thread = true), Base.front(args)...)
end

function def_outer_reduct_types!(ls::LoopSet)
  for or ∈ ls.outer_reductions
    op = operations(ls)[or]
    pushpreamble!(ls, Expr(:(=), outer_reduct_init_typename(op), typeof_expr(op)))
  end
end
"""
    @_turbo

This macro mostly exists for debugging/testing purposes. It does not support many of the use cases of [`@turbo`](@ref).
It emits loops directly, rather than punting to an `@generated` function, meaning it doesn't have access to type
information when generating code or analyzing the loops, often leading to bad performance.

This macro accepts the `inline` and `unroll` keyword arguments like `@turbo`, but ignores the `check_empty` argument.
"""
macro _turbo(q)
  q = macroexpand(__module__, q)
  ls = LoopSet(q, __module__)
  set_hw!(ls)
  def_outer_reduct_types!(ls)
  esc(Expr(:block, ls.prepreamble, lower_and_split_loops(ls, -1)))
end
macro _turbo(arg, q)
  @assert q.head === :for
  q = macroexpand(__module__, q)
  inline, check_empty, u₁, u₂, v =
    check_macro_kwarg(arg, false, false, zero(Int8), zero(Int8), zero(Int8), 1, 0, true)
  ls = LoopSet(q, __module__)
  set_hw!(ls)
  def_outer_reduct_types!(ls)
  esc(Expr(:block, ls.prepreamble, lower(ls, u₁ % Int, u₂ % Int, v % Int, -1)))
end

"""
    @turbo_debug

Returns a `LoopSet` object instead of evaluating the loops. Useful for debugging and introspection.
"""
macro turbo_debug(q)
  q = macroexpand(__module__, q)
  ls = LoopSet(q, __module__)
  esc(LoopVectorization.setup_call_debug(ls))
end

# define aliases
const var"@avx" = var"@turbo"
const var"@avxt" = var"@tturbo"
const var"@avx_debug" = var"@turbo_debug"
