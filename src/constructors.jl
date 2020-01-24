
### This file contains convenience functions for constructing LoopSets.

function Base.copyto!(ls::LoopSet, q::Expr)
    q.head === :for || throw("Expression must be a for loop.")
    add_loop!(ls, q)
end

function add_ci_call!(q::Expr, f, args, syms, i, mod = nothing)
    call = Expr(:call, f)
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
        elseif f === GlobalRef(Base, :materialize!)
            add_ci_call!(ex, lv(:vmaterialize!), ciₙargs, syms, n, mod)
        elseif f === GlobalRef(Base, :materialize)
            add_ci_call!(ex, lv(:vmaterialize), ciₙargs, syms, n, mod)
        else
            add_ci_call!(ex, f, ciₙargs, syms, n)
        end
    end
    ex
end

function LoopSet(q::Expr, mod::Symbol = :LoopVectorization)
    q = SIMDPirates.contract_pass(q)
    ls = LoopSet(mod)
    copyto!(ls, q)
    resize!(ls.loop_order, num_loops(ls))
    ls
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

The macro models the set of nested loops, and chooses a 

It may also apply to broadcasts:

```jldoctest
julia> a = rand(100);

julia> b = @avx exp.(2 .* a);

julia> c = similar(b);

julia> @avx @. c = exp(2a);

julia> b ≈ c 
true
```

"""
macro avx(q)
    mod = Symbol(__module__)
    q2 = if q.head === :for
        setup_call(LoopSet(q, mod))
    else# assume broadcast
        substitute_broadcast(q, mod)
    end
    esc(q2)
end

function check_inline(arg)
    a1 = (arg.args[1])::Symbol
    a1 === :inline || return nothing
    (arg.args[2])::Bool
end
function check_tile(arg)
    a1 = (arg.args[1])::Symbol
    a1 === :tile || return nothing
    U = convert(Int8, tup.args[1])
    T = convert(Int8, tup.args[2])
    U, T
end
function check_unroll(arg)
    a1 = (arg.args[1])::Symbol
    a1 === :unroll || return nothing
    convert(Int8, arg.args[2])
end
function check_macro_kwarg(arg, inline::Int8 = Int8(2), U::Int8 = zero(Int8), T::Int8 = zero(Int8))
    @assert arg.head === :(=)
    i = check_inline(arg)
    if i !== nothing
        inline = i ? Int8(2) : Int8(-1)
    else
        u = check_unroll(arg)
        if u !== nothing
            U = u
            T = Int8(-1)
        else
            U, T = check_tile(arg)
        end
    end
    inline, U, T
end
macro avx(arg, q)
    @assert q.head === :for
    @assert arg.head === :(=)
    inline, U, T = check_macro_kwarg(arg)
    esc(setup_call(LoopSet(q, Symbol(__module__)), inline, U, T))
end
macro avx(arg1, arg2, q)
    @assert q.head === :for
    inline, U, T = check_macro_kwarg(arg1)
    inline, U, T = check_macro_kwarg(arg2, inline, U, T)
    esc(setup_call(LoopSet(q, Symbol(__module__)), inline, U, T))
end



macro _avx(q)
    esc(lower(LoopSet(q, Symbol(__module__))))
end
macro _avx(arg, q)
    @assert q.head === :for
    inline, U, T = check_macro_kwarg(arg)
    esc(lower(LoopSet(q, Symbol(__module__)), U, T))
end



