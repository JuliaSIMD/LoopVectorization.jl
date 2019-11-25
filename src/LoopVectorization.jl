module LoopVectorization

using VectorizationBase, SIMDPirates, SLEEFPirates, MacroTools
using VectorizationBase: REGISTER_SIZE, extract_data, num_vector_load_expr
using SIMDPirates: VECTOR_SYMBOLS
using MacroTools: @capture, prewalk, postwalk

export vectorizable, @vectorize, @vvectorize

function isdense end #

const SLEEFPiratesDict = Dict{Symbol,Tuple{Symbol,Symbol}}(
    :sin => (:SLEEFPirates, :sin_fast),
    :sinpi => (:SLEEFPirates, :sinpi),
    :cos => (:SLEEFPirates, :cos_fast),
    :cospi => (:SLEEFPirates, :cospi),
    :tan => (:SLEEFPirates, :tan_fast),
    :log => (:SLEEFPirates, :log_fast),
    :log10 => (:SLEEFPirates, :log10),
    :log2 => (:SLEEFPirates, :log2),
    :log1p => (:SLEEFPirates, :log1p),
    :exp => (:SLEEFPirates, :exp),
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
    :zero => (:SIMDPirates, :vzero)
)





@noinline function _spirate(ex, dict, macro_escape = true, mod = :LoopVectorization)
    ex = postwalk(ex) do x
        if @capture(x, a_ += b_)
            return :($a = $mod.vadd($a, $b))
        elseif @capture(x, a_ -= b_)
            return :($a = $mod.vsub($a, $b))
        elseif @capture(x, a_ *= b_)
            return :($a = $mod.vmul($a, $b))
        elseif @capture(x, a_ /= b_)
            return :($a = $mod.vdiv($a, $b))
        elseif @capture(x, Base.FastMath.add_fast(a__))
            return :($mod.vadd($(a...)))
        elseif @capture(x, Base.FastMath.sub_fast(a__))
            return :($mod.vsub($(a...)))
        elseif @capture(x, Base.FastMath.mul_fast(a__))
            return :($mod.vmul($(a...)))
        elseif @capture(x, Base.FastMath.div_fast(a__))
            return :($mod.vfdiv($(a...)))
        elseif @capture(x, a_ / sqrt(b_))
            return :($a * $mod.rsqrt($b))
        elseif @capture(x, inv(sqrt(a_)))
            return :($mod.rsqrt($a))
        elseif @capture(x, @horner a__)
            return SIMDPirates.horner(a...)
        elseif @capture(x, Base.Math.muladd(a_, b_, c_))
            return :( $mod.vmuladd($a, $b, $c) )
        elseif isa(x, Symbol) && !occursin("@", string(x))
            vec_mod, vec_sym = get(dict, x, (:not_found,:not_found))
            if vec_sym != :not_found
                return :($mod.$vec_mod.$vec_sym)
            else
                vec_sym = get(VECTOR_SYMBOLS, x, :not_found)
                return vec_sym == :not_found ? x : :($mod.SIMDPirates.$(vec_sym))
            end
        else
            return x
        end
    end
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
        vectorize_body(N, Float32, uf, n, body, vecdict, VType, mod)
    elseif Tsym == :Float64
        vectorize_body(N, Float64, uf, n, body, vecdict, VType, mod)
    else
        throw("Type $Tsym is not supported.")
    end
end
@noinline function vectorize_body(N, T::DataType, unroll_factor::Int, n, body, vecdict = SLEEFPiratesDict, VType = SVec, gcpreserve::Bool = true, mod = :LoopVectorization)
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
        push!(main_body.args,
            _vectorloads!(main_body, q, indexed_expressions, reduction_symbols, loaded_exprs, V, W, T, loop_constants_quote, loop_constants_dict, b;
                            itersym = itersym, declared_iter_sym = n, VectorizationDict = vecdict, mod = mod)
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
    if unroll_factor == 1
        for ((sym,op),gsym) ∈ reduction_symbols
            if op == :+ || op == :-
                pushfirst!(q.args, :($gsym = $mod.vbroadcast($V,zero($T))))
            elseif op == :* || op == :/
                pushfirst!(q.args, :($gsym = $mod.vbroadcast($V,one($T))))
            end
            if op == :+
                push!(q.args, :($sym = Base.FastMath.add_fast($sym, $mod.vsum($gsym))))
            elseif op == :-
                push!(q.args, :($sym = Base.FastMath.sub_fast($sym, $mod.vsum($gsym))))
            elseif op == :*
                push!(q.args, :($sym = Base.FastMath.mul_fast($sym, $mod.SIMDPirates.vprod($gsym))))
            elseif op == :/
                push!(q.args, :($sym = Base.FastMath.div_fast($sym, $mod.SIMDPirates.vprod($gsym))))
            end
        end
    else
        for ((sym,op),gsym_base) ∈ reduction_symbols
            for uf ∈ 0:unroll_factor-1
                gsym = Symbol(gsym_base, :_, uf)
                if op == :+ || op == :-
                    pushfirst!(q.args, :($gsym = $mod.vbroadcast($V,zero($T))))
                elseif op == :* || op == :/
                    pushfirst!(q.args, :($gsym = $mod.vbroadcast($V,one($T))))
                end
            end
            func = ((op == :*) | (op == :/)) ? :($mod.vmul) : :($mod.vadd)
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
            if op == :+
                push!(q.args, :($sym = Base.FastMath.add_fast($sym, $mod.vsum($gsym))))
            elseif op == :-
                push!(q.args, :($sym = Base.FastMath.sub_fast($sym, $mod.vsum($gsym))))
            elseif op == :*
                push!(q.args, :($sym = Base.FastMath.mul_fast($sym, $mod.SIMDPirates.vprod($gsym))))
            elseif op == :/
                push!(q.args, :($sym = Base.FastMath.div_fast($sym, $mod.SIMDPirates.vprod($gsym))))
            end
        end
    end
    push!(q.args, nothing)
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

@noinline function add_masks(expr, masksym, reduction_symbols, default_module = :LoopVectorization)
    # println("Called add masks!")
    # postwalk(expr) do x
    prewalk(expr) do x
        if @capture(x, M_.vstore!(args__))
            M === nothing && (M = default_module)
            return :($M.vstore!($(args...), $masksym))
        elseif @capture(x, M_.vload(args__))
            M === nothing && (M = default_module)
            return :($M.vload($(args...), $masksym))
        # We mask the reductions, because the odds of them getting contaminated and therefore poisoning the results seems too great
        # for reductions to be practical. If what we're vectorizing is simple enough not to worry about contamination...then
        # it ought to be simple enough so we don't need @vectorize.
        elseif @capture(x, reductionA_ = M_.vadd(reductionA_, B_ ) ) || @capture(x, reductionA_ = M_.vadd(B_, reductionA_ ) ) || @capture(x, reductionA_ = vadd(reductionA_, B_ ) ) || @capture(x, reductionA_ = vadd(B_, reductionA_ ) )
            M === nothing && (M = default_module)
            return :( $reductionA = $M.vifelse($masksym, $M.vadd($reductionA, $B), $reductionA) )
        elseif @capture(x, reductionA_ = M_.vmul(reductionA_, B_ ) ) || @capture(x, reductionA_ = M_.vmul(B_, reductionA_ ) ) ||  @capture(x, reductionA_ = vmul(reductionA_, B_ ) ) || @capture(x, reductionA_ = vmul(B_, reductionA_ ) )
            M === nothing && (M = default_module)
            return :( $reductionA = $M.vifelse($masksym, $M.vmul($reductionA, $B), $reductionA) )
        elseif @capture(x, reductionA_ = M_.f_(B_, C_, reductionA_) ) ||  @capture(x, reductionA_ = f_(B_, C_, reductionA_) )
            M === nothing && (M = default_module)
            return :( $reductionA = $M.vifelse($masksym, $M.$f($B, $C, $reductionA), $reductionA) )
        # elseif @capture(x, reductionA_ = M_.vfnmadd(B_, C_, reductionA_ ) ) || @capture(x, reductionA_ = vfnmadd(B_, C_, reductionA_ ) )
            # M === nothing && (M = default_module)
            # return :( $reductionA = $M.vifelse($masksym, $M.vfnmadd($B, $C, $reductionA), $reductionA) )
        elseif @capture(x, reductionA_ = M_.f_(reductionA_, B_ ) ) || @capture(x, reductionA_ = f_(reductionA_, B_ ) )
            M === nothing && (M = default_module)
            return :( $reductionA = $M.vifelse($masksym, $M.$f($reductionA, $B), $reductionA) )
#        elseif @capture(x, reductionA_ = M_.vmul(reductionA_, B_ ) )
            # M === nothing && (M = :(LoopVectorization.SIMDPirates))
#            return :( $reductionA = $M.vifelse($masksym, $M.vmul($reductionA, $B), $reductionA) )
        else
            return x
        end
    end
end


@noinline function _vectorloads!(main_body, pre_quote, indexed_expressions, reduction_symbols, loaded_exprs, V, W, VET, loop_constants_quote, loop_constants_dict, expr;
                            itersym = :iter, declared_iter_sym = nothing, VectorizationDict = SLEEFPiratesDict, mod = :LoopVectorization)
    _spirate(prewalk(expr) do x
        # @show x
        # @show main_body
        if @capture(x, A_[i__] += B_)
             x = :($A[$(i...)] = $B + $A[$(i...)])
        elseif @capture(x, A_[i__] -= B_)
             x = :($A[$(i...)] = $A[$(i...)] - $B)
        elseif @capture(x, A_[i__] *= B_)
             x = :($A[$(i...)] = $B * $A[$(i...)])
        elseif @capture(x, A_[i__] /= B_)
             x = :($A[$(i...)] = $A[$(i...)] / $B)
        end
        if @capture(x, A_[i_] = B_) || @capture(x, setindex!(A_, B_, i_))
            # println("Made it.") 
            if A ∉ keys(indexed_expressions)
                # pA = esc(gensym(A))
                # pA = esc(Symbol(:p,A))
                pA = gensym(Symbol(:p,A))
                indexed_expressions[A] = pA
            else
                pA = indexed_expressions[A]
            end
            if i == declared_iter_sym
                return :($mod.vstore!($pA, $B, $itersym))
            elseif isa(i, Expr)
                contains_itersym, i2 = subsymbol(i, declared_iter_sym, itersym)
                return :($mod.vstore!($pA, $B, $i2))
            else
                return :($mod.vstore!($pA, $B, $i))
            end
        elseif @capture(x, A_[i_,j_] = B_) || @capture(x, setindex!(A_, B_, i_, j_))
            if A ∉ keys(indexed_expressions)
                pA = gensym(Symbol(:p, A))
                indexed_expressions[A] = pA
            else
                pA = indexed_expressions[A]
            end
            sym = gensym(Symbol(pA, :_, i))
            if i == declared_iter_sym
                # then i gives the row number
                # ej gives the number of columns the setindex! is shifted
                ej = isa(j, Number) ? j - 1 : :($j - 1)
                stridexpr = :($mod.LoopVectorization.stride_row($A))
                if stridexpr ∈ keys(loop_constants_dict)
                    stridesym = loop_constants_dict[stridexpr]
                else
                    stridesym = gensym(:stride)
                    push!(loop_constants_quote.args, :( $stridesym = $stridexpr ))
                    loop_constants_dict[stridexpr] = stridesym
                end
                return :($mod.vstore!($pA, $B, $itersym + $ej*$stridesym))
            else
                throw("Indexing columns with vectorized loop variable is not supported.")
            end
        elseif (@capture(x, A_ += B_) || @capture(x, A_ = A_ + B_) || @capture(x, A_ = B_ + A_)) && A isa Symbol
            # @show A, typeof(A)
            gA = get!(() -> gensym(A), reduction_symbols, (A, :+))
            return :( $gA = $mod.vadd($gA, $B ))
        elseif (@capture(x, A_ -= B_) || @capture(x, A_ = A_ - B_)) && A isa Symbol
            # @show A, typeof(A)
            gA = get!(() -> gensym(A), reduction_symbols, (A, :-))
            return :( $gA = $mod.vadd($gA, $B ))
        elseif (@capture(x, A_ *= B_) || @capture(x, A_ = A_ * B_) || @capture(x, A_ = B_ * A_)) && A isa Symbol
            # @show A, typeof(A)
            gA = get!(() -> gensym(A), reduction_symbols, (A, :*))
            return :( $gA = $mod.vmul($gA, $B ))
        elseif (@capture(x, A_ /= B_) || @capture(x, A_ = A_ / B_)) && A isa Symbol
            # @show A, typeof(A)
            gA = get!(() -> gensym(A), reduction_symbols, (A, :/))
            return :( $gA = $mod.vmul($gA, $B ))
        elseif @capture(x, A_[i_]) || @capture(x, getindex(A_, i_))
            if A ∉ keys(indexed_expressions)
                # pA = esc(gensym(A))
                # pA = esc(Symbol(:p,A))
                pA = gensym(Symbol(:p,A))
                indexed_expressions[A] = pA
            else
                pA = indexed_expressions[A]
            end
            ## check to see if we are to do a vector load or a broadcast
            if i == declared_iter_sym
                load_expr = :($mod.vload($V, $pA, $itersym ))
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
            if load_expr ∈ keys(loaded_exprs)
                sym = loaded_exprs[load_expr]
            else
                sym = gensym(Symbol(pA, :_i))
                loaded_exprs[load_expr] = sym
                push!(main_body.args, :($sym = $load_expr))
            end
            # return the symbol we assigned the load to.
            return sym
        elseif @capture(x, A_[i_, j_]) || @capture(x, getindex(A_, i_, j_))
            if A ∉ keys(indexed_expressions)
                # pA = esc(gensym(A))
                # pA = esc(Symbol(:p,A))
                pA = gensym(Symbol(:p,A))
                indexed_expressions[A] = pA
            else
                pA = indexed_expressions[A]
            end
            ej = isa(j, Number) ?  j - 1 : :($j - 1)
            if i == declared_iter_sym
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
            if load_expr ∈ keys(loaded_exprs)
                sym = loaded_exprs[load_expr]
            else
                sym = gensym(Symbol(pA, :_, i))
                loaded_exprs[load_expr] = sym
                push!(main_body.args, :($sym = $load_expr))
            end
            # return the symbol we assigned the load to.
            return sym
        elseif @capture(x, A_[i_,:] .= B_)
            ## Capture if there are multiple assignments...
            if A ∉ keys(indexed_expressions)
                pA = gensym(Symbol(:p,A))
                indexed_expressions[A] = pA
            else
                pA = indexed_expressions[A]
            end
            if i == declared_iter_sym
                isym = itersym
            else
                isym = i
            end
            br = gensym(:B)
            br2 = gensym(:B)
            coliter = gensym(:j)
            stridexpr = :($mod.LoopVectorization.stride_row($A))
            if stridexpr ∈ keys(loop_constants_dict)
                stridesym = loop_constants_dict[stridexpr]
            else
                stridesym = gensym(:stride)
                push!(loop_constants_quote.args, :( $stridesym = $stridexpr ))
                loop_constants_dict[stridexpr] = stridesym
            end
            expr = quote
                $br = $mod.LoopVectorization.extract_data.($B)
                for $coliter ∈ 0:length($br)-1
                    @inbounds $mod.vstore!($pA, getindex($br,1+$coliter), $isym + $stridesym * $coliter)
                end
            end
            return expr
        elseif @capture(x, zero(T_))
            return :(zero($V))
        elseif @capture(x, one(T_))
            return :(one($V))
        elseif @capture(x, B_ ? A_ : C_)
            return :($mod.vifelse($B, $A, $C))
        elseif x == declared_iter_sym
            isymvec = gensym(itersym)
            push!(pre_quote.args, :($isymvec = SVec($(Expr(:tuple, [:(Core.VecElement{$VET}($(w-W))) for w ∈ 1:W]...)))))
            push!(main_body.args, :($isymvec = $mod.vadd($isymvec, vbroadcast($V, $W)) ))
            return isymvec
        else
            return x
        end
    end, VectorizationDict, false, mod) # macro_escape = false
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


"""
Arguments are
@vectorize Type UnrollFactor forloop

The default type is Float64, and default UnrollFactor is 1 (no unrolling).
"""

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

end # module
