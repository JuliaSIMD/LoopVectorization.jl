module LoopVectorization

using VectorizationBase, SIMDPirates, SLEEFPirates, MacroTools, Parameters
using VectorizationBase: REGISTER_SIZE, REGISTER_COUNT, extract_data, num_vector_load_expr, mask
using SIMDPirates: VECTOR_SYMBOLS, evadd, evmul, vrange, reduced_add, reduced_prod
using Base.Broadcast: Broadcasted, DefaultArrayStyle
using LinearAlgebra: Adjoint
using MacroTools: prewalk, postwalk


export LowDimArray, stridedpointer, vectorizable,
    @vectorize, @vvectorize, @avx, ∗

function isdense end #

const SLEEFPiratesDict = Dict{Symbol,Tuple{Symbol,Symbol}}(
    :sin => (:SLEEFPirates, :sin_fast),
    :sinpi => (:SLEEFPirates, :sinpi),
    :cos => (:SLEEFPirates, :cos_fast),
    :cospi => (:SLEEFPirates, :cospi),
    :tan => (:SLEEFPirates, :tan_fast),
    # :log => (:SLEEFPirates, :log_fast),
    :log => (:SIMDPirates, :vlog),
    :log10 => (:SLEEFPirates, :log10),
    :log2 => (:SLEEFPirates, :log2),
    :log1p => (:SLEEFPirates, :log1p),
    # :exp => (:SLEEFPirates, :exp),
    :exp => (:SIMDPirates, :vexp),
    :exp2 => (:SLEEFPirates, :exp2),
    :exp10 => (:SLEEFPirates, :exp10),
    :expm1 => (:SLEEFPirates, :expm1),
    :inv => (:SIMDPirates, :vinv), # faster than sqrt_fast
    :sqrt => (:SIMDPirates, :sqrt), # faster than sqrt_fast
    :rsqrt => (:SIMDPirates, :rsqrt),
    :cbrt => (:SLEEFPirates, :cbrt_fast),
    :asin => (:SLEEFPirates, :asin_fast),
    :acos => (:SLEEFPirates, :acos_fast),
    :atan => (:SLEEFPirates, :atan_fast),
    :sinh => (:SLEEFPirates, :sinh),
    :cosh => (:SLEEFPirates, :cosh),
    :tanh => (:SLEEFPirates, :tanh),
    :asinh => (:SLEEFPirates, :asinh),
    :acosh => (:SLEEFPirates, :acosh),
    :atanh => (:SLEEFPirates, :atanh),
    # :erf => :(SLEEFPirates.erf),
    # :erfc => :(SLEEFPirates.erfc),
    # :gamma => :(SLEEFPirates.gamma),
    # :lgamma => :(SLEEFPirates.lgamma),
    :trunc => (:SLEEFPirates, :trunc),
    :floor => (:SLEEFPirates, :floor),
    :ceil => (:SIMDPirates, :ceil),
    :abs => (:SIMDPirates, :vabs),
    :sincos => (:SLEEFPirates, :sincos_fast),
    # :pow => (:SLEEFPirates, :pow_fast),
    :^ => (:SLEEFPirates, :pow_fast),
    # :sincospi => (:SLEEFPirates, :sincospi_fast),
    # :pow => (:SLEEFPirates, :pow),
    # :hypot => (:SLEEFPirates, :hypot_fast),
    :mod => (:SLEEFPirates, :mod),
    # :copysign => :copysign
    :one => (:SIMDPirates, :vone),
    :zero => (:SIMDPirates, :vzero),
    :erf => (:SIMDPirates, :verf)
)



# @noinline function _spirate(ex, dict, macro_escape = true, mod = :LoopVectorization)
#     ex = postwalk(ex) do x
#         if @capture(x, a_ += b_)
#             return :($a = $mod.vadd($a, $b))
#         elseif @capture(x, a_ -= b_)
#             return :($a = $mod.vsub($a, $b))
#         elseif @capture(x, a_ *= b_)
#             return :($a = $mod.vmul($a, $b))
#         elseif @capture(x, a_ /= b_)
#             return :($a = $mod.vdiv($a, $b))
#         elseif @capture(x, Base.FastMath.add_fast(a__))
#             return :($mod.vadd($(a...)))
#         elseif @capture(x, Base.FastMath.sub_fast(a__))
#             return :($mod.vsub($(a...)))
#         elseif @capture(x, Base.FastMath.mul_fast(a__))
#             return :($mod.vmul($(a...)))
#         elseif @capture(x, Base.FastMath.div_fast(a__))
#             return :($mod.vfdiv($(a...)))
#         elseif @capture(x, a_ / sqrt(b_))
#             return :($a * $mod.rsqrt($b))
#         elseif @capture(x, inv(sqrt(a_)))
#             return :($mod.rsqrt($a))
#         elseif @capture(x, @horner a__)
#             return SIMDPirates.horner(a...)
#         elseif @capture(x, Base.Math.muladd(a_, b_, c_))
#             return :( $mod.vmuladd($a, $b, $c) )
#         elseif isa(x, Symbol) && !occursin("@", string(x))
#             vec_mod, vec_sym = get(dict, x, (:not_found,:not_found))
#             if vec_sym != :not_found
#                 return :($mod.$vec_mod.$vec_sym)
#             else
#                 vec_sym = get(VECTOR_SYMBOLS, x, :not_found)
#                 return vec_sym == :not_found ? x : :($mod.SIMDPirates.$(vec_sym))
#             end
#         else
#             return x
#         end
#     end
#     macro_escape ? esc(ex) : ex
# end

@noinline function _spirate(ex, dict, macro_escape = true, mod = :LoopVectorization)
    ex = postwalk(ex) do x
        if x isa Symbol
            vec_mod, vec_sym = get(dict, x) do
                mod, get(VECTOR_SYMBOLS, x) do
                    x
                end
            end
            return x === vec_sym ? x : Expr(:(.), vec_mod === mod ? mod : Expr(:(.), mod, QuoteNode(vec_mod)), QuoteNode(vec_sym))
        end
        x isa Expr || return x
        xexpr::Expr = x
        # if xexpr.head === :macrocall && first(xexpr.args) === Symbol("@horner")
            # return SIMDPirates.horner(@view(xexpr.args[3:end])...)
        # end
        xexpr.head === :call || return x
        f = first(xexpr.args)
        if f == :(Base.FastMath.add_fast)
            vf = :vadd
        elseif f == :(Base.FastMath.sub_fast)
            vf = :vsub
        elseif f == :(Base.FastMath.mul_fast)
            vf = :vmul
        elseif f == :(Base.FastMath.div_fast)
            vf = :vfdiv
        elseif f == :(Base.FastMath.sqrt)
            vf = :vsqrt
        elseif f == :(Base.Math.muladd)
            vf = :vmuladd
        else
            return xexpr
        end
        return Expr(:call, Expr(:(.), mod, QuoteNode(vf)), @view(x.args[2:end])...)
    end
    # println(ex)
    macro_escape ? esc(ex) : ex
end







"""
Returns the strides necessary to iterate across rows.
Needs `@inferred` testing / that the compiler optimizes it away
whenever size(A) is known at compile time. Seems to be the case for Julia 1.1.
"""
@inline stride_row(A::AbstractArray) = size(A,1)

function replace_syms_i(expr, set, i)
    postwalk(expr) do ex
        if ex isa Symbol && ex ∈ set
            return Symbol(ex, :_, i)
        else
            return ex
        end
    end
end

@noinline function vectorize_body(N, Tsym::Symbol, uf, n, body, vecdict = SLEEFPiratesDict, VType = SVec, gcpreserve::Bool = true , mod = :LoopVectorization)
    if Tsym == :Float32
        vectorize_body(N, Float32, uf, n, body, vecdict, VType, gcpreserve, mod)
    elseif Tsym == :Float64
        vectorize_body(N, Float64, uf, n, body, vecdict, VType, gcpreserve, mod)
    else
        throw("Type $Tsym is not supported.")
    end
end
@noinline function vectorize_body(
    N, ::Type{T}, unroll_factor::Int, n::Symbol, body,
    vecdict::Dict{Symbol,Tuple{Symbol,Symbol}} = SLEEFPiratesDict,
    @nospecialize(VType = SVec), gcpreserve::Bool = true, mod = :LoopVectorization
) where {T}
    # unroll_factor == 1 || throw("Only unroll factor of 1 is currently supported. Was set to $unroll_factor.")
    T_size = sizeof(T)
    if isa(N, Integer)
        W, Wshift = VectorizationBase.pick_vector_width_shift(N, T)
        Nsym = N
    else
        W, Wshift = VectorizationBase.pick_vector_width_shift(T)
        Nsym = gensym(:N)
    end
    if !(N isa Integer) && unroll_factor > 1 # We will force the unroll to be a power of two
        log2unroll = max(0,VectorizationBase.intlog2(unroll_factor))
        unroll_factor = 1 << log2unroll  
    else
        log2unroll = 0
    end
    WT = W * T_size
    V = VType{W,T}
    vectorize_body(N, Nsym, VType{W,T}, unroll_factor, n, body, vecdict, gcpreserve, Wshift, log2unroll, mod)
end
@noinline function vectorize_body(
    N, Nsym, ::Type{V}, unroll_factor, n, body, vecdict, gcpreserve, Wshift, log2unroll, mod
) where {W,T,V <: Union{SVec{W,T},Vec{W,T}}}
    indexed_expressions = Dict{Symbol,Symbol}() # Symbol, gensymbol

    itersym = gensym(:i)
    # walk the expression, searching for all get index patterns.
    # these will be replaced with
    main_body = quote end
    reduction_symbols = Dict{Tuple{Symbol,Symbol},Symbol}()
    loaded_exprs = Dict{Expr,Symbol}()
    loop_constants_dict = Dict{Expr,Symbol}()
    loop_constants_quote = quote end

    ### Here we define unrolled_loop count, full loop count, and rem loop
    if N isa Integer
        Q, r = divrem(N, unroll_factor*W)
        Qp1W = (Q+1) << Wshift
        if N % Qp1W == 0
            Q, r = Q + 1, 0
            unroll_factor = N ÷ Qp1W
        end
        q = quote end
        # loop_max_expr = Q - 1
        loop_max_expr = Q
        remr = r >>> Wshift
        r &= (W - 1)
    else
        Qsym = gensym(:Q)
        remsym = gensym(:rem)
        remr = gensym(:remreps)
        q = quote
            $Nsym = $N
            ($Qsym, $remsym) = $(num_vector_load_expr(:($mod.LoopVectorization), Nsym, W<<log2unroll))
        end
        if unroll_factor > 1
            push!(q.args, :($remr = $remsym >>> $Wshift))
            push!(q.args, :($remsym &= $(W-1)))
        end
        loop_max_expr = Qsym
    end
    # @show T
    for b ∈ body
        b = macroexpand(LoopVectorization, b)
        ## body preamble must define indexed symbols
        ## we only need that for loads.
        dicts = (indexed_expressions, reduction_symbols, loaded_exprs, loop_constants_dict)
        push!(
            main_body.args,
            _vectorloads!(
                main_body, q, dicts, V, loop_constants_quote, b;
                itersym = itersym, declared_iter_sym = n, VectorizationDict = vecdict, mod = mod
            )
        )# |> x -> (@show(x), _pirate(x)))
    end
    # @show main_body

    for (sym, psym) ∈ indexed_expressions
        push!(q.args, :( $psym = $mod.vectorizable($sym) ))
    end
    push!(q.args, loop_constants_quote)
    push!(q.args, :($itersym = 0))
    unrolled_loop_body_expr = quote end
    unrolled_loop_body_iter = quote
        $main_body
        $itersym += $W
    end
    if unroll_factor == 1
        push!(unrolled_loop_body_expr.args, unrolled_loop_body_iter)
    else
        ulb = unrolled_loop_body_iter
        rep_syms = Set(values(reduction_symbols))
        unrolled_loop_body_iter = replace_syms_i(ulb, rep_syms, 0)
        push!(unrolled_loop_body_expr.args, unrolled_loop_body_iter)
        for u in 1:unroll_factor-1
            push!(unrolled_loop_body_expr.args, replace_syms_i(ulb, rep_syms, u))
        end
    end
    unadjitersym = gensym(:unadjitersym)
    if loop_max_expr isa Integer && loop_max_expr <= 1
        loop_max_expr == 1 && push!(q.args, unrolled_loop_body_expr)
    else
        loop_quote = quote
            for $unadjitersym ∈ 1:$loop_max_expr
                $unrolled_loop_body_expr
            end
        end
        push!(q.args, loop_quote)
    end
    if N isa Integer
        for _ in 1:remr
            push!(q.args, unrolled_loop_body_iter)
        end
    else
        if unroll_factor > 1
            unrolled_remquote = quote
                for $unadjitersym in 1:$remr
                    $unrolled_loop_body_iter
                end
            end
            push!(q.args, unrolled_remquote)
        end
    end
    if !(N isa Integer) || r > 0
        masksym = gensym(:mask)
        masked_loop_body = add_masks(unrolled_loop_body_iter, masksym, reduction_symbols, mod)
        if N isa Integer
            push!(q.args, quote
                $masksym = $(VectorizationBase.mask(T, r))
                # $itersym = $(N - r)
                $masked_loop_body
            end)
        else
            push!(q.args, quote
                if $remsym > 0
                    $masksym = $mod.VectorizationBase.mask(Val{$W}(), $remsym)
                    # $itersym = ($Nsym - $remsym)
                    $masked_loop_body
                end
            end)
        end
    end
    ### now we walk the body to look for reductions
    add_reductions!(q, V, reduction_symbols, unroll_factor, mod)
    # display(q)
    # We are using pointers, so better add a GC.@preserve.
    # gcpreserve = true
    # gcpreserve = false
    if gcpreserve
        return quote
            $(Expr(:macrocall,
        Expr(:., :GC, QuoteNode(Symbol("@preserve"))),
            LineNumberNode(@__LINE__), (keys(indexed_expressions))..., q
                   ))
            nothing
        end
    else
        return q
    end
end

function add_reductions!(q, ::Type{V}, reduction_symbols, unroll_factor, mod) where {W,T,V <: Union{SVec{W,T},Vec{W,T}}}
    if unroll_factor == 1
        for ((sym,op),gsym) ∈ reduction_symbols
            if op === :+ || op === :-
                pushfirst!(q.args, :($gsym = $mod.vbroadcast($V,zero($T))))
            elseif op === :* || op === :/
                pushfirst!(q.args, :($gsym = $mod.vbroadcast($V,one($T))))
            end
            if op === :+
                push!(q.args, :($sym = $mod.SIMDPirates.reduced_add($sym, $gsym)))
            elseif op === :-
                push!(q.args, :($sym = Base.FastMath.sub_fast($sym, $mod.vsum($gsym))))
            elseif op === :*
                push!(q.args, :($sym = $mod.SIMDPirates.reduced_prod($sym, $gsym)))
            elseif op === :/
                push!(q.args, :($sym = Base.FastMath.div_fast($sym, $mod.SIMDPirates.vprod($gsym))))
            end
        end
    else
        for ((sym,op),gsym_base) ∈ reduction_symbols
            for uf ∈ 0:unroll_factor-1
                gsym = Symbol(gsym_base, :_, uf)
                if op === :+ || op === :-
                    pushfirst!(q.args, :($gsym = $mod.vbroadcast($V,zero($T))))
                elseif op === :* || op === :/
                    pushfirst!(q.args, :($gsym = $mod.vbroadcast($V,one($T))))
                end
            end
            func = ((op === :*) | (op === :/)) ? :($mod.evmul) : :($mod.evadd)
            uf_new = unroll_factor
            while uf_new > 1
                uf_new, uf_prev = uf_new >> 1, uf_new
                for uf ∈ 0:uf_new - 1 # reduce half divisible by two
                    push!(q.args, Expr(:(=), Symbol(gsym_base, :_, uf), Expr(:call, func, Symbol(gsym_base, :_, 2uf), Symbol(gsym_base, :_, 2uf + 1))))
                end
                uf_firstrem = 2uf_new
                for uf ∈ uf_firstrem:uf_prev - 1
                    push!(q.args, Expr(:(=), Symbol(gsym_base, :_, uf - uf_firstrem), Expr(:call, func, Symbol(gsym_base, :_, uf - uf_firstrem), Symbol(gsym_base, :_, uf))))
                end
            end
            gsym = Symbol(gsym_base, :_, 0)
            if op === :+
                push!(q.args, :($sym = Base.FastMath.add_fast($sym, $mod.vsum($gsym))))
            elseif op === :-
                push!(q.args, :($sym = Base.FastMath.sub_fast($sym, $mod.vsum($gsym))))
            elseif op === :*
                push!(q.args, :($sym = Base.FastMath.mul_fast($sym, $mod.SIMDPirates.vprod($gsym))))
            elseif op === :/
                push!(q.args, :($sym = Base.FastMath.div_fast($sym, $mod.SIMDPirates.vprod($gsym))))
            end
        end
    end
    push!(q.args, nothing)
    nothing
end

function insert_mask(x, masksym, reduction_symbols, default_module = :LoopVectorization)
    x isa Expr || return x
    local fs::Symbol, mf::Expr, f::Union{Symbol,Expr}, call::Expr, a::Symbol
    if x.head === :(=) # check for reductions
        x.args[2] isa Expr || return x
        # @show x
        a = x.args[1]
        call = x.args[2]
        f = first(call.args)
        for i ∈ 2:length(call.args)
            if call.args[i] === a
                if f isa Symbol
                    call = Expr(:call, Expr(:., default_module, QuoteNode(f)), @view(call.args[2:end])...)
                end
                return Expr(:(=), a, Expr(:call, Expr(:., default_module, QuoteNode(:vifelse)), masksym, call, a))
            end
        end
        return x
    elseif x.head === :call # check for vload or vstore
        f = first(x.args)::Union{Symbol,Expr}
        if f isa Symbol
            fs = f
            (fs === :vload || fs === :vstore!) || return x
            mf = Expr(:., default_module, QuoteNode(f))
        elseif f isa Expr
            # @show f
            fs = f.args[2].value
            (fs === :vload || fs === :vstore!) || return x
            mf = f
        end
        return Expr(:call, mf, @view(x.args[2:end])..., masksym)
    else
        x
    end
end

@noinline function add_masks(expr, masksym, reduction_symbols, default_module = :LoopVectorization)
    # println("Called add masks!")
    # postwalk(expr) do x
    prewalk(expr) do x
        insert_mask(x, masksym, reduction_symbols, default_module)
    end
end

function vectorize_assign_linear_index(A, B, i, indexed_expressions, itersym, declared_iter_sym, mod)
    pA = get!(indexed_expressions, A) do
        gensym(Symbol(:p,A))
    end
    ind = if i == declared_iter_sym
        itersym
    elseif isa(i, Expr)
        last(subsymbol(i, declared_iter_sym, itersym))
    else
        i
    end
    Expr(:call, Expr(:., mod, QuoteNode(:vstore!)), pA, B, ind)
end
function vectorize_assign_cartesian_index(A, B, i, j, indexed_expressions, itersym, declared_iter_sym, mod)
    pA = get!(indexed_expressions, A) do
        gensym(Symbol(:p,A))
    end
    sym = gensym(Symbol(pA, :_, i))
    if i == declared_iter_sym
        # then i gives the row number
        # ej gives the number of columns the setindex! is shifted
        ej = isa(j, Number) ? j - 1 : Expr(:call, :-, j, 1)
        stridexpr = :($mod.LoopVectorization.stride_row($A))
        if stridexpr ∈ keys(loop_constants_dict)
            stridesym = loop_constants_dict[stridexpr]
        else
            stridesym = gensym(:stride)
            push!(loop_constants_quote.args, :( $stridesym = $stridexpr ))
            loop_constants_dict[stridexpr] = stridesym
        end
        return Expr(:call, Expr(:., mod, QuoteNode(:vstore!)), pA, B, Expr(:call, :+, itersym, Expr(:call, :*, ej, stridesym)))
        # return :($mod.vstore!($pA, $B, $itersym + $ej*$stridesym))
    else
        throw("Indexing columns with vectorized loop variable is not supported.")
    end
end
function vectorize_linear_index!(main_body, loaded_exprs, indexed_expressions, A, i, itersym, declared_iter_sym, mod, V)
    pA = get!(indexed_expressions, A) do
        gensym(Symbol(:p,A))
    end
    ## check to see if we are to do a vector load or a broadcast
    if i === declared_iter_sym
        load_expr = Expr(:call, Expr(:., mod, QuoteNode(:vload)), V, pA, itersym )
    elseif isa(i, Expr)
        contains_itersym, i2 = subsymbol(i, declared_iter_sym, itersym)
        if contains_itersym
            load_expr = :($mod.vload($V, $pA, $i2 ))
        else
            load_expr = :($mod.vbroadcast($V, $pA - 1 + $i))
        end
    else
        load_expr = :($mod.vbroadcast($V, $pA - 1 + $i))
    end
    # performs a CSE on load expressions
    get!(loaded_exprs, load_expr) do
        sym = gensym(Symbol(pA, :_i))
        push!(main_body.args, Expr(:(=), sym, load_expr))
        sym
    end
end
function vectorize_cartesian_index!(main_body, loaded_exprs, indexed_expressions, A, i, j, itersym, declared_iter_sym, mod, V)
    pA = get!(indexed_expressions, A) do
        gensym(Symbol(:p,A))
    end
    ej = isa(j, Number) ?  j - 1 : Expr(:(-), j, 1)
    if i === declared_iter_sym
        stridexpr = :($mod.LoopVectorization.stride_row($A))
        if stridexpr ∈ keys(loop_constants_dict)
            stridesym = loop_constants_dict[stridexpr]
        else
            stridesym = gensym(:stride)
            push!(loop_constants_quote.args, :( $stridesym = $stridexpr ))
            loop_constants_dict[stridexpr] = stridesym
        end
        load_expr = :($mod.vload($V, $pA, $itersym + $ej*$stridesym))
    elseif j == declared_iter_sym
        throw("Indexing columns with vectorized loop variable is not supported.")
    else
        # when loading something not indexed by the loop variable,
        # we assume that the intension is to broadcast
        stridexpr = :($mod.LoopVectorization.stride_row($A))
        if stridexpr ∈ keys(loop_constants_dict)
            stridesym = loop_constants_dict[stridexpr]
        else
            stridesym = gensym(:stride)
            push!(loop_constants_quote.args, :( $stridesym = $stridexpr ))
            loop_constants_dict[stridexpr] = stridesym
        end
        # added -1 because i is not the declared itersym, therefore the row number is presumably 1-indexed.
        load_expr = :($mod.vbroadcast($V, LoopVectorization.VectorizationBase.load($pA + $i - 1 + $ej*$stridesym)))
    end
    # performs a CSE on load expressions
    get!(loaded_exprs, load_expr) do
        sym = gensym(Symbol(pA, :_, i))
        push!(main_body.args, :($sym = $load_expr))
        sym
    end
end
function vectorize_broadcast_across_columns(indexed_expressions, loop_constants_quote, loop_constants_dict, B, A, i, itersym, declared_iter_sym, mod)
    ## Capture if there are multiple assignments...
    pA = get!(indexed_expressions, A) do
        gensym(Symbol(:p,A))
    end
    if i === declared_iter_sym
        isym = itersym
    else
        isym = i
    end
    br = gensym(:B)
    coliter = gensym(:j)
    stridexpr = :($mod.LoopVectorization.stride_row($A))
    stridesym = get!(loop_constants_dict, stridexpr) do
        stridesym = gensym(:stride)
        push!(loop_constants_quote.args, Expr(:(=), stridesym, stridexpr ))
        stridesym
    end
    quote
        $br = $mod.LoopVectorization.extract_data.($B)
        for $coliter ∈ 0:length($br)-1
            @inbounds $mod.vstore!($pA, getindex($br,1+$coliter), $isym + $stridesym * $coliter)
        end
    end
end
function vectorload!(
    dicts, main_body, loop_constants_quote,
    x::Expr, ::Type{V}, itersym, declared_iter_sym, mod
) where {W,T,V <: Union{Vec{W,T},SVec{W,T}}}
    (indexed_expressions, reduction_symbols, loaded_exprs, loop_constants_dict) = dicts
    if x.head === :+=
        x = Expr(:(=), first(x.args), Expr(:call, :+, x.args...))
    elseif x.head === :-=
        x = Expr(:(=), first(x.args), Expr(:call, :-, x.args...))
    elseif x.head === :*=
        x = Expr(:(=), first(x.args), Expr(:call, :*, x.args...))
    elseif x.head === :/=
        x = Expr(:(=), first(x.args), Expr(:call, :/, x.args...))
    end
    local Assigned::Union{Expr,Symbol}, Assignedexpr::Expr, Assignedsym::Symbol, f::Union{Expr,Symbol}, fs::Symbol, B::Union{Symbol,Expr}, Bexpr::Expr
    if x.head === :(=)
        Assigned = first(x.args)
        if Assigned isa Symbol
            B = x.args[2]
            if B isa Symbol
                return x
            else
                Assignedsym = Assigned
                Bexpr = B
                if Bexpr.head === :call
                    f = first(Bexpr.args)
                    if f isa Symbol
                        fs = f
                        if fs === :+ || fs === :*
                            for i ∈ 2:length(Bexpr.args)
                                if Bexpr.args[i] === Assignedsym # this is a reduction
                                    gA = get!(() -> gensym(Assignedsym), reduction_symbols, (Assignedsym, fs))
                                    vf = fs === :+ ? :vadd : :vmul
                                    call = Expr(:call, Expr(:., mod, QuoteNode(vf)), gA )
                                    for j ∈ 2:length(Bexpr.args)
                                        j == i && continue
                                        push!(call.args, Bexpr.args[j])
                                    end
                                    return Expr(:(=), gA, call)
                                end
                            end
                        elseif (fs === :- || fs === :/) && Bexpr.args[2] === Assignedsym
                            gA = get!(() -> gensym(Assignedsym), reduction_symbols, (Assignedsym, fs))
                            return Expr(:(=), gA, Expr(:call, Expr(:., mod, QuoteNode(fs === :- ? :vadd : :vmul)), gA, Bexpr.args[3] ))
                        end
                        return x
                    else
                        return x
                    end
                else
                    return x
                end
            end
        else
            Assignedexpr = Assigned
            if Assignedexpr.head === :ref
                ninds = length(Assignedexpr.args) -1
                if ninds == 1
                    return vectorize_assign_linear_index(
                        first(Assignedexpr.args), last(x.args), last(Assignedexpr.args),
                        indexed_expressions, itersym, declared_iter_sym, mod
                    )
                elseif ninds == 2
                    return vectorize_assign_cartesian_index(
                        first(Assignedexpr.args), last(x.args), Assignedexpr.args[2], Assignedexpr.args[3],
                        indexed_expressions, itersym, declared_iter_sym, mod
                    )
                else
                    throw("Currently only supports up to 2 indices for some reason.")
                end
            end
        end
    elseif x.head === :ref
        if length(x.args) == 2
            return vectorize_linear_index!(main_body, loaded_exprs, indexed_expressions, x.args[1], x.args[2], itersym, declared_iter_sym, mod, V)
        elseif length(x.args) == 3
            return vectorize_cartesian_index!(main_body, loaded_exprs, indexed_expressions, x.args[1], x.args[2], x.args[3], itersym, declared_iter_sym, mod, V)
        else
            throw("Currently only supports up to 2 indices for some reason.")
        end
    elseif x.head === :call
        f = first(x.args)
        if f === :setindex!
            ninds = length(Assignedexpr.args) - 3
            if ninds == 1
                return vectorize_assign_linear_index(
                    x.args[2], x.args[3], x.args[4],
                    indexed_expressions, itersym, declared_iter_sym, mod
                )
            elseif ninds == 2
                return vectorize_assign_cartesian_index(
                    x.args[2], x.args[3], x.args[4], x.args[5],
                    indexed_expressions, itersym, declared_iter_sym, mod
                )
            else
                throw("Currently only supports up to 2 indices for some reason.")
            end
        elseif f === :getindex
            ninds = length(Assignedexpr.args) - 2
            if ninds == 1
                return vectorize_linear_index!(main_body, loaded_exprs, indexed_expressions, x.args[2], x.args[3], itersym, declared_iter_sym, mod, V)
            elseif ninds == 2
                return vectorize_cartesian_index!(main_body, loaded_exprs, indexed_expressions, x.args[2], x.args[3], x.args[4], itersym, declared_iter_sym, mod, V)
            else
                throw("Currently only supports up to 2 indices for some reason.")
            end
        elseif f === :zero
            return Expr(:call, Expr(:(.), mod, QuoteNode(:vbroadcast)), V, zero(T))
        elseif f === :one
            return Expr(:call, Expr(:(.), mod, QuoteNode(:vbroadcast)), V, one(T))
        else
            return x
        end
    elseif x.head === :if
        return Expr(:call, Expr(:(.), mod, QuoteNode(:vifelse)), x.args...)
    elseif x.head === :(.=) && x.args[1].args[3] === :(:)
        return vectorize_broadcast_across_columns(indexed_expressions, loop_constants_dict, B, A, i, itersym, declared_iter_sym, mod)
    end
    x
end

@noinline function _vectorloads!(
    main_body, pre_quote, dicts, ::Type{V}, loop_constants_quote, expr;
    itersym = :iter, declared_iter_sym = nothing, VectorizationDict = SLEEFPiratesDict, mod = :LoopVectorization
) where {W,T,V <: Union{Vec{W,T},SVec{W,T}}}
    q = prewalk(expr) do x
        # @show x
        if x isa Symbol
            if x === declared_iter_sym
                isymvec = gensym(itersym)
                if V == SVec
                    push!(pre_quote.args, :($isymvec = SVec($(Expr(:tuple, [:(Core.VecElement{$T}($(w-W))) for w ∈ 1:W]...)))))
                else
                    push!(pre_quote.args, :($isymvec = $(Expr(:tuple, [:(Core.VecElement{$T}($(w-W))) for w ∈ 1:W]...))))
                end
                push!(main_body.args, :($isymvec = $mod.vadd($isymvec, vbroadcast($V, $W)) ))
                return isymvec
            else
                return x
            end
        end
        x isa Expr || return x
        vectorload!(
            dicts, main_body, loop_constants_quote, x, V, itersym, declared_iter_sym, mod            
        )       
    end
    _spirate(q, VectorizationDict, false, mod) # macro_escape = false
end

"""
subsymbol(expr, i, j)
substitute symbol i with symbol j in expr
Returns true if a substitution was made, false otherwise.
"""
@noinline function subsymbol(expr::Expr, i, j)
    subbed = false
    expr = postwalk(expr) do ex
        if ex == i
            subbed = true
            return j
        else
            return ex
        end
    end
    subbed, expr
end

# function loop_components(expr::Expr)
#     expr.head === :for || throw("Macro must be applied to a for loop.")
#     iterdef = expr.args[1]
#     itersym = iterdef.args[1]
#     iterrange = iterdef.args[2]
#     @assert iterrange isa Expr
#     @assert length(expr.args) == 2
#     body = expr.args[2]
#     iterlength = if iterrange.head === :call
#         if iterrange.args[1] === :(:)
#             if iterrange.args[2] == 1
#                 iterrange.args[3]
#             else
#                 Expr(:(-), iterrange.args[3], iterrange.args[2])
#             end
#         elseif iterrange.args[1] === :eachindex
#             if length(iterrange.args) == 2
#                 Expr(:call, :length, iterrange.args[2])
#             else
#                 il = Expr(:call, :min)
#                 for i ∈ 2:length(iterrange.args)
#                     push!(il.args, Expr(:call, :length, iterrange.args[i]))
#                 end
#                 il
#             end
#         else
#             throw("could not match loop expression.")
#         end
#     end
#     @show iterdef, itersym
#     iterlength, itersym, body
# end


# # Arguments are
# # @vectorize Type UnrollFactor forloop

# # The default type is Float64, and default UnrollFactor is 1 (no unrolling).


# for vec ∈ (false,true)
#     if vec
#         V = Vec
#         macroname = :vvectorize
#     else
#         V = SVec
#         macroname = :vectorize
#     end
#     for gcpreserve ∈ (true,false)
#         if !gcpreserve
#             macroname = Symbol(macroname, :_unsafe)
#         end
#         @eval macro $macroname(expr)
#             iterlength, itersym, body = loop_components(expr)
#             esc(vectorize_body(iterlength, Float64, 1, itersym, body, SLEEFPiratesDict, $V, $gcpreserve))
#         end
#         @eval macro $macroname(type, expr)
#             iterlength, itersym, body = loop_components(expr)
#             esc(vectorize_body(iterlength, type, 1, itersym, body, SLEEFPiratesDict, $V, $gcpreserve))
#         end
#         @eval macro $macroname(unroll_factor::Integer, expr)
#             iterlength, itersym, body = loop_components(expr)
#             esc(vectorize_body(iterlength, Float64, unroll_factor, itersym, body, SLEEFPiratesDict, $V, $gcpreserve))
#         end
#         @eval macro $macroname(type, unroll_factor::Integer, expr)
#             iterlength, itersym, body = loop_components(expr)
#             esc(vectorize_body(iterlength, type, unroll_factor, itersym, body, SLEEFPiratesDict, $V, $gcpreserve))
#         end
#         @eval macro $macroname(type, mod::Union{Symbol,Module}, expr)
#             iterlength, itersym, body = loop_components(expr)
#             esc(vectorize_body(iterlength, type, 1, itersym, body, SLEEFPiratesDict, $V, $gcpreserve, mod))
#         end
#         @eval macro $macroname(type, mod::Union{Symbol,Module}, unroll_factor::Integer, expr)
#             iterlength, itersym, body = loop_components(expr)
#             esc(vectorize_body(iterlength, type, unroll_factor, itersym, body, SLEEFPiratesDict, $V, $gcpreserve, mod))
#         end
#     end
# end

for vec ∈ (false,true)
    if vec
        V = Vec
        macroname = :vvectorize
    else
        V = SVec
        macroname = :vectorize
    end
    for gcpreserve ∈ (true,false)
        if !gcpreserve
            macroname = Symbol(macroname, :_unsafe)
        end
        @eval macro $macroname(expr)
            if @capture(expr, for n_ ∈ 1:N_ body__ end)
                q = vectorize_body(N, Float64, 1, n, body, SLEEFPiratesDict, $V, $gcpreserve)
            elseif @capture(expr, for n_ ∈ eachindex(A_) body__ end)
                q = vectorize_body(:(length($A)), Float64, 1, n, body, SLEEFPiratesDict, $V, $gcpreserve)
            elseif @capture(expr, for n_ ∈ eachindex(args__) body__ end)
                q = vectorize_body(:(min($([:(length($a)) for a ∈ args]...))), Float64, 1, n, body, SLEEFPiratesDict, $V, $gcpreserve)
            else
                throw("Could not match loop expression.")
            end
            esc(q)
        end
        @eval macro $macroname(type, expr)
            if @capture(expr, for n_ ∈ 1:N_ body__ end)
                q = vectorize_body(N, type, 1, n, body, SLEEFPiratesDict, $V, $gcpreserve)
            elseif @capture(expr, for n_ ∈ eachindex(A_) body__ end)
                q = vectorize_body(:(length($A)), type, 1, n, body, SLEEFPiratesDict, $V, $gcpreserve)
            elseif @capture(expr, for n_ ∈ eachindex(args__) body__ end)
                q = vectorize_body(:(min($([:(length($a)) for a ∈ args]...))), type, 1, n, body, SLEEFPiratesDict, $V, $gcpreserve)
            else
                throw("Could not match loop expression.")
            end
            esc(q)
        end
        @eval macro $macroname(unroll_factor::Integer, expr)
            if @capture(expr, for n_ ∈ 1:N_ body__ end)
                q = vectorize_body(N, Float64, unroll_factor, n, body, SLEEFPiratesDict, $V, $gcpreserve)
            elseif @capture(expr, for n_ ∈ eachindex(A_) body__ end)
                q = vectorize_body(:(length($A)), Float64, unroll_factor, n, body, SLEEFPiratesDict, $V, $gcpreserve)
            elseif @capture(expr, for n_ ∈ eachindex(args__) body__ end)
                q = vectorize_body(:(min($([:(length($a)) for a ∈ args]...))), Float64, unroll_factor, n, body, SLEEFPiratesDict, $V, $gcpreserve)
            else
                throw("Could not match loop expression.")
            end
            esc(q)
        end
        @eval macro $macroname(type, unroll_factor::Integer, expr)
            if @capture(expr, for n_ ∈ 1:N_ body__ end)
                q = vectorize_body(N, type, unroll_factor, n, body, SLEEFPiratesDict, $V, $gcpreserve)
            elseif @capture(expr, for n_ ∈ eachindex(A_) body__ end)
                q = vectorize_body(:(length($A)), type, unroll_factor, n, body, SLEEFPiratesDict, $V, $gcpreserve)
            elseif @capture(expr, for n_ ∈ eachindex(args__) body__ end)
                q = vectorize_body(:(min($([:(length($a)) for a ∈ args]...))), type, unroll_factor, n, body, SLEEFPiratesDict, $V, $gcpreserve)
            else
                throw("Could not match loop expression.")
            end
            esc(q)
        end
        @eval macro $macroname(type, mod::Union{Symbol,Module}, expr)
            if @capture(expr, for n_ ∈ 1:N_ body__ end)
                q = vectorize_body(N, type, 1, n, body, SLEEFPiratesDict, $V, $gcpreserve, mod)
            elseif @capture(expr, for n_ ∈ eachindex(A_) body__ end)
                q = vectorize_body(:(length($A)), type, 1, n, body, SLEEFPiratesDict, $V, $gcpreserve, mod)
            elseif @capture(expr, for n_ ∈ eachindex(args__) body__ end)
                q = vectorize_body(:(min($([:(length($a)) for a ∈ args]...))), type, 1, n, body, SLEEFPiratesDict, $V, $gcpreserve, mod)
            else
                throw("Could not match loop expression.")
            end
            esc(q)
        end
        @eval macro $macroname(type, mod::Union{Symbol,Module}, unroll_factor::Integer, expr)
            if @capture(expr, for n_ ∈ 1:N_ body__ end)
                q = vectorize_body(N, type, unroll_factor, n, body, SLEEFPiratesDict, $V, mod)
            elseif @capture(expr, for n_ ∈ eachindex(A_) body__ end)
                q = vectorize_body(:(length($A)), type, unroll_factor, n, body, SLEEFPiratesDict, $V, mod)
            elseif @capture(expr, for n_ ∈ eachindex(args__) body__ end)
                q = vectorize_body(:(min($([:(length($a)) for a ∈ args]...))), type, unroll_factor, n, body, SLEEFPiratesDict, $V, $gcpreserve, mod)
            else
                throw("Could not match loop expression.")
            end
            esc(q)
        end
    end
end

include("costs.jl")
include("operations.jl")
include("graphs.jl")
include("broadcast.jl")
include("determinestrategy.jl")
include("lowering.jl")
include("constructors.jl")
include("precompile.jl")
_precompile_()

end # module
