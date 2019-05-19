module LoopVectorization

using VectorizationBase, SIMDPirates, SLEEFPirates, MacroTools
using VectorizationBase: REGISTER_SIZE, extract_data
using SIMDPirates: VECTOR_SYMBOLS
using MacroTools: @capture, prewalk, postwalk

export vectorizable, @vectorize



const SLEEFPiratesDict = Dict{Symbol,Expr}(
    :sin => :(SLEEFPirates.sin_fast),
    :sinpi => :(SLEEFPirates.sinpi),
    :cos => :(SLEEFPirates.cos_fast),
    :cospi => :(SLEEFPirates.cospi),
    :tan => :(SLEEFPirates.tan_fast),
    :log => :(SLEEFPirates.log_fast),
    :log10 => :(SLEEFPirates.log10),
    :log2 => :(SLEEFPirates.log2),
    :log1p => :(SLEEFPirates.log1p),
    :exp => :(SLEEFPirates.exp),
    :exp2 => :(SLEEFPirates.exp2),
    :exp10 => :(SLEEFPirates.exp10),
    :expm1 => :(SLEEFPirates.expm1),
    :sqrt => :(SLEEFPirates.sqrt), # faster than sqrt_fast
    :rsqrt => :(LoopVectorization.SIMDPirates.rsqrt),
    :cbrt => :(SLEEFPirates.cbrt_fast),
    :asin => :(SLEEFPirates.asin_fast),
    :acos => :(SLEEFPirates.acos_fast),
    :atan => :(SLEEFPirates.atan_fast),
    :sinh => :(SLEEFPirates.sinh),
    :cosh => :(SLEEFPirates.cosh),
    :tanh => :(SLEEFPirates.tanh),
    :asinh => :(SLEEFPirates.asinh),
    :acosh => :(SLEEFPirates.acosh),
    :atanh => :(SLEEFPirates.atanh),
    # :erf => :(SLEEFPirates.erf),
    # :erfc => :(SLEEFPirates.erfc),
    # :gamma => :(SLEEFPirates.gamma),
    # :lgamma => :(SLEEFPirates.lgamma),
    :trunc => :(SLEEFPirates.trunc),
    :floor => :(SLEEFPirates.floor),
    :ceil => :(SLEEFPirates.ceil),
    :abs => :(SLEEFPirates.abs),
    :sincos => :(SLEEFPirates.sincos_fast),
    # :sincospi => :(SLEEFPirates.sincospi_fast),
    # :pow => :(SLEEFPirates.pow),
    # :hypot => :(SLEEFPirates.hypot_fast),
    :mod => :(SLEEFPirates.mod)
    # :copysign => :copysign
)





function _spirate(ex, dict, macro_escape = true)
    ex = postwalk(ex) do x
        # @show x
        # if @capture(x, LoopVectorization.SIMDPirates.vadd(LoopVectorization.SIMDPirates.vmul(a_, b_), c_)) || @capture(x, LoopVectorization.SIMDPirates.vadd(c_, LoopVectorization.SIMDPirates.vmul(a_, b_)))
        #     return :(LoopVectorization.SIMDPirates.vmuladd($a, $b, $c))
        # elseif @capture(x, LoopVectorization.SIMDPirates.vadd(LoopVectorization.SIMDPirates.vmul(a_, b_), LoopVectorization.SIMDPirates.vmul(c_, d_), e_)) || @capture(x, LoopVectorization.SIMDPirates.vadd(LoopVectorization.SIMDPirates.vmul(a_, b_), e_, LoopVectorization.SIMDPirates.vmul(c_, d_))) || @capture(x, LoopVectorization.SIMDPirates.vadd(e_, LoopVectorization.SIMDPirates.vmul(a_, b_), LoopVectorization.SIMDPirates.vmul(c_, d_)))
        #     return :(LoopVectorization.SIMDPirates.vmuladd($a, $b, LoopVectorization.SIMDPirates.vmuladd($c, $d, $e)))
        # elseif @capture(x, LoopVectorization.SIMDPirates.vadd(LoopVectorization.SIMDPirates.vmul(b_, c_), LoopVectorization.SIMDPirates.vmul(d_, e_), LoopVectorization.SIMDPirates.vmul(f_, g_), a_)) ||
        #         @capture(x, LoopVectorization.SIMDPirates.vadd(LoopVectorization.SIMDPirates.vmul(b_, c_), LoopVectorization.SIMDPirates.vmul(d_, e_), a_, LoopVectorization.SIMDPirates.vmul(f_, g_))) ||
        #         @capture(x, LoopVectorization.SIMDPirates.vadd(LoopVectorization.SIMDPirates.vmul(b_, c_), a_, LoopVectorization.SIMDPirates.vmul(d_, e_), LoopVectorization.SIMDPirates.vmul(f_, g_))) ||
        #         @capture(x, LoopVectorization.SIMDPirates.vadd(a_, LoopVectorization.SIMDPirates.vmul(b_, c_), LoopVectorization.SIMDPirates.vmul(d_, e_), LoopVectorization.SIMDPirates.vmul(f_, g_)))
        #     return :(LoopVectorization.SIMDPirates.vmuladd($g, $f, LoopVectorization.SIMDPirates.vmuladd($e, $d, LoopVectorization.SIMDPirates.vmuladd($c, $b, $a))))
        # elseif @capture(x, a_ * b_ + c_ - c_) || @capture(x, c_ + a_ * b_ - c_) || @capture(x, a_ * b_ - c_ + c_) || @capture(x, - c_ + a_ * b_ + c_)
        #     return :(LoopVectorization.SIMDPirates.vmul($a, $b))
        # elseif @capture(x, a_ * b_ + c_ - d_) || @capture(x, c_ + a_ * b_ - d_) || @capture(x, a_ * b_ - d_ + c_) || @capture(x, - d_ + a_ * b_ + c_) || @capture(x, LoopVectorization.SIMDPirates.vsub(LoopVectorization.SIMDPirates.vmuladd(a_, b_, c_), d_))
        #     return :(LoopVectorization.SIMDPirates.vmuladd($a, $b, LoopVectorization.SIMDPirates.vsub($c, $d)))
        # elseif @capture(x, a_ += b_)
        if @capture(x, a_ += b_)
            return :($a = LoopVectorization.SIMDPirates.vadd($a, $b))
        elseif @capture(x, a_ -= b_)
            return :($a = LoopVectorization.SIMDPirates.vsub($a, $b))
        elseif @capture(x, a_ *= b_)
            return :($a = LoopVectorization.SIMDPirates.vmul($a, $b))
        elseif @capture(x, a_ /= b_)
            return :($a = LoopVectorization.SIMDPirates.vdiv($a, $b))
        elseif @capture(x, a_ / sqrt(b_))
            return :($a * rsqrt($b))
        elseif @capture(x, inv(sqrt(a_)))
            return :(rsqrt($a))
        elseif @capture(x, @horner a__)
            return horner(a...)
        elseif @capture(x, Base.Math.muladd(a_, b_, c_))
            return :( LoopVectorization.SIMDPirates.vmuladd($a, $b, $c) )
        elseif isa(x, Symbol) && !occursin("@", string(x))
            vec_sym = get(dict, x, :not_found)
            if vec_sym != :not_found
                return vec_sym
            else
                vec_sym = get(VECTOR_SYMBOLS, x, :not_found)
                return vec_sym == :not_found ? x : :(LoopVectorization.SIMDPirates.$(vec_sym))
            end
        else
            return x
        end
    end
    macro_escape ? esc(ex) : ex
end







# mask_expr(W, r) = :($(Expr(:tuple, [i > r ? Core.VecElement{Bool}(false) : Core.VecElement{Bool}(true) for i ∈ 1:W]...)))

"""
Returns the strides necessary to iterate across rows.
Needs `@inferred` testing / that the compiler optimizes it away
whenever size(A) is known at compile time. Seems to be the case for Julia 1.1.
"""
@inline stride_row(A::AbstractArray) = size(A,1)
@inline function num_row_strides(A::AbstractArray)
    s = size(A)
    N = s[2]
    for i ∈ 3:length(s)
        N *= s[i]
    end
    N
end
@inline function stride_row_iter(A::AbstractArray)
    N = num_row_strides(A)
    stride = stride_row(A)
    ntuple(i -> (i-1) * stride, Val(N))
end

function mask_expr(W, remsym::Symbol)
    if W <= 8
        m = :((one(UInt8) << $remsym) - one(UInt8))
    elseif W <= 16
        m = :((one(UInt16) << $remsym) - one(UInt16))
    elseif W <= 32
        m = :((one(UInt32) << $remsym) - one(UInt32))
    elseif W <= 64
        m = :((one(UInt64) << $remsym) - one(UInt64))
    elseif W <= 128
        m = :((one(UInt128) << $remsym) - one(UInt128))
    else
        throw("A mask of length $W > 128? Are you sure you want to do that?")
    end
    m
end
mask(rem::Integer) = (one(UInt) << rem) - one(UInt)
function create_mask(W, r)
    if W <= 8
        return UInt8(2)^r-UInt8(1)
    elseif W <= 16
        return UInt16(2)^r-UInt16(1)
    elseif W <= 32
        return UInt32(2)^r-UInt32(1)
    elseif W <= 64
        return UInt64(2)^r-UInt64(1)
    else #W <= 128
        return UInt128(2)^r-UInt128(1)
    end
end

function vectorize_body(N, Tsym::Symbol, uf, n, body, vecdict = SLEEFPiratesDict, VType = SVec)
    if Tsym == :Float32
        vectorize_body(N, Float32, uf, n, body, vecdict, VType)
    elseif Tsym == :Float64
        vectorize_body(N, Float64, uf, n, body, vecdict, VType)
    elseif Tsym == :ComplexF32
        vectorize_body(N, ComplexF32, uf, n, body, vecdict, VType)
    elseif Tsym == :ComplexF64
        vectorize_body(N, ComplexF64, uf, n, body, vecdict, VType)
    else
        throw("Type $Tsym is not supported.")
    end
end
function vectorize_body(N, T::DataType, unroll_factor, n, body, vecdict = SLEEFPiratesDict, VType = SVec)
    # unroll_factor == 1 || throw("Only unroll factor of 1 is currently supported. Was set to $unroll_factor.")
    T_size = sizeof(T)
    if isa(N, Integer)
        W, Wshift = VectorizationBase.pick_vector_width_shift(N, T)
        Nsym = N
    else
        W, Wshift = VectorizationBase.pick_vector_width_shift(T)
        Nsym = gensym(:N)
    end
    log2unroll = max(1,VectorizationBase.intlog2(unroll_factor))
    W *= unroll_factor
    Wshift += log2unroll
    # @show W, REGISTER_SIZE, T_size
    # @show T
    WT = W * T_size
    V = VType{W,T}

    # @show body

    # body = _pirate(body)

    # indexed_expressions = Dict{Symbol,Expr}()
    indexed_expressions = Dict{Symbol,Symbol}() # Symbol, gensymbol

    itersym = gensym(:i)
    # walk the expression, searching for all get index patterns.
    # these will be replaced with
    # Plan: definition of q will create vectorizables

    main_body = quote end
    reduction_symbols = Dict{Tuple{Symbol,Symbol},Symbol}()
    loaded_exprs = Dict{Expr,Symbol}()
    loop_constants_dict = Dict{Expr,Symbol}()
    loop_constants_quote = quote end


    if isa(N, Integer)
        Q, r = divrem(N, W)
        q = quote end
        loop_max_expr = Q - 1
    else
        Qsym = gensym(:Q)
        remsym = gensym(:rem)
        q = quote
            $Nsym = $N
            ($Qsym, $remsym) = $(num_vector_load_expr(:LoopVectorization, N, W))
            # $loop_constants_quote
        end
        loop_max_expr = :($Qsym-1)
    end
    # @show T
    for b ∈ body
        b = nexprs_expansion(b)
        ## body preamble must define indexed symbols
        ## we only need that for loads.
        push!(main_body.args,
            _vectorloads!(main_body, q, indexed_expressions, reduction_symbols, loaded_exprs, V, W, T, loop_constants_quote, loop_constants_dict, b;
                            itersym = itersym, declared_iter_sym = n, VectorizationDict = vecdict)
        )# |> x -> (@show(x), _pirate(x)))
    end
    # @show main_body

    for (sym, psym) ∈ indexed_expressions
        push!(q.args, :( $psym = vectorizable($sym) ))
    end
    push!(q.args, loop_constants_quote)

    unadjitersym = gensym(:unadjitersym)
    if !isa(loop_max_expr, Integer) || loop_max_expr >= 0
        push!(q.args,
        quote
            for $unadjitersym ∈ 0:$loop_max_expr
                $itersym = $W * $unadjitersym
                $main_body
            end
        end)
    end

    if !isa(N, Integer) || r > 0
        masksym = gensym(:mask)
        masked_loop_body = add_masks(main_body, masksym, reduction_symbols)
        if isa(N, Integer)
            push!(q.args, quote
                $masksym = $(create_mask(W, r))
                $itersym = $(N - r)
                $masked_loop_body
            end)
        else
            push!(q.args, quote
                if $remsym > 0
                    $masksym = $(mask_expr(W, remsym))
                    $itersym = ($Nsym - $remsym)
                    $masked_loop_body
                end
            end)
        end
    end


    ### now we walk the body to look for reductions
    for ((sym,op),gsym) ∈ reduction_symbols
        if op == :+ || op == :-
            pushfirst!(q.args, :($gsym = LoopVectorization.SIMDPirates.vbroadcast($V,zero($T))))
        elseif op == :* || op == :/
            pushfirst!(q.args, :($gsym = LoopVectorization.SIMDPirates.vbroadcast($V,one($T))))
        end
        if op == :+
            # push!(q.args, :(@show $sym, $gsym))
            push!(q.args, :(@fastmath $sym = $sym + LoopVectorization.SIMDPirates.vsum($gsym)))
            # push!(q.args, :(@show $sym, $gsym))
        elseif op == :-
            push!(q.args, :(@fastmath $sym = $sym - LoopVectorization.SIMDPirates.vsum($gsym)))
        elseif op == :*
            push!(q.args, :(@fastmath $sym = $sym * LoopVectorization.SIMDPirates.vprod($gsym)))
        elseif op == :/
            push!(q.args, :(@fastmath $sym = $sym / LoopVectorization.SIMDPirates.vprod($gsym)))
        end
    end

    # display(q)
    # We are using pointers, so better add a GC.@preserve.
    Expr(:macrocall,
        Expr(:., :GC, QuoteNode(Symbol("@preserve"))),
            LineNumberNode(@__LINE__), (keys(indexed_expressions))..., q
    )
    # q
end

function add_masks(expr, masksym, reduction_symbols)
    postwalk(expr) do x
        if @capture(x, LoopVectorization.SIMDPirates.vstore!(ptr_, V_))
            return :(LoopVectorization.SIMDPirates.vstore!($ptr, $V, $masksym))
        elseif @capture(x, LoopVectorization.SIMDPirates.vload(V_, ptr_))
            return :(LoopVectorization.SIMDPirates.vload($V, $ptr, $masksym))
        # We mask the reductions, because the odds of them getting contaminated and therefore poisoning the results seems too great
        # for reductions to be practical. If what we're vectorizing is simple enough not to worry about contamination...then
        # it ought to be simple enough so we don't need @vectorize.
        elseif @capture(x, reductionA_ = LoopVectorization.SIMDPirates.vadd(reductionA_, B_ ) )
            return :( $reductionA = SIMDPirates.vifelse($masksym, LoopVectorization.SIMDPirates.vadd($reductionA, $B), $reductionA) )
        elseif @capture(x, reductionA_ = LoopVectorization.SIMDPirates.vmul(reductionA_, B_ ) )
            return :( $reductionA = SIMDPirates.vifelse($masksym, LoopVectorization.SIMDPirates.vmul($reductionA, $B), $reductionA) )
        else
            return x
        end
    end
end

# function _vectorloads(V, expr; itersym = :iter, declared_iter_sym = nothing, VectorizationDict = SLEEFPiratesDict)
#
#
#     # body = _pirate(body)
#
#     # indexed_expressions = Dict{Symbol,Expr}()
#     indexed_expressions = Dict{Symbol,Symbol}() # Symbol, gensymbol
#
#     main_body = quote end
#     reduction_symbols = Dict{Tuple{Symbol,Symbol},Symbol}()
#     loaded_exprs = Dict{Expr,Symbol}()
#     loop_constants_dict = Dict{Expr,Symbol}()
#     loop_constants_quote = quote end
#
#     push!(main_body.args,
#         _vectorloads!(main_body, indexed_expressions, reduction_symbols, loaded_exprs, V, loop_constants_quote, loop_constants_dict, expr;
#             itersym = itersym, declared_iter_sym = declared_iter_sym, VectorizationDict = VectorizationDict)
#     )
#     main_body
# end

function nexprs_expansion(expr)
    prewalk(expr) do x
        if @capture(x, @nexprs N_ ex_) || @capture(x, Base.Cartesian.@nexprs N_ ex_)
            # println("Macroexpanding x:", x)
            # @show ex
            # mx = Expr(:escape, Expr(:block, Any[ Base.Cartesian.inlineanonymous(ex,i) for i = 1:N ]...))
            mx = Expr(:block, Any[ Base.Cartesian.inlineanonymous(ex,i) for i = 1:N ]...)
            # println("Macroexpanded x:", mx)
            return mx
        else
            # @show x
            return x
        end
    end
end

function _vectorloads!(main_body, pre_quote, indexed_expressions, reduction_symbols, loaded_exprs, V, W, VET, loop_constants_quote, loop_constants_dict, expr;
                            itersym = :iter, declared_iter_sym = nothing, VectorizationDict = SLEEFPiratesDict)
    _spirate(prewalk(expr) do x
        # @show x
        # @show main_body
        if @capture(x, A_[i_] = B_)
            if A ∉ keys(indexed_expressions)
                # pA = esc(gensym(A))
                # pA = esc(Symbol(:p,A))
                pA = gensym(Symbol(:p,A))
                indexed_expressions[A] = pA
            else
                pA = indexed_expressions[A]
            end
            if i == declared_iter_sym
                return :(LoopVectorization.SIMDPirates.vstore!($pA + $itersym, $B))
            elseif isa(i, Expr)
                contains_itersym, i2 = subsymbol(i, declared_iter_sym, itersym)
                return :(LoopVectorization.SIMDPirates.vstore!($pA + $i2, $B))
            else
                return :(LoopVectorization.SIMDPirates.vstore!($pA + $i, $B))
            end
        elseif @capture(x, A_[i_,j_] = B_)
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
                stridexpr = :(LoopVectorization.stride_row($A))
                if stridexpr ∈ keys(loop_constants_dict)
                    stridesym = loop_constants_dict[stridexpr]
                else
                    stridesym = gensym(:stride)
                    push!(loop_constants_quote.args, :( $stridesym = $stridexpr ))
                    loop_constants_dict[stridexpr] = stridesym
                end
                return :(LoopVectorization.SIMDPirates.vstore!($pA + $itersym + $ej*$stridesym, $B))
            else
                throw("Indexing columns with vectorized loop variable is not supported.")
            end
        elseif (@capture(x, A_ += B_) || @capture(x, A_ = A_ + B_) || @capture(x, A_ = B_ + A_)) && A isa Symbol
            # @show A, typeof(A)
            gA = get!(() -> gensym(A), reduction_symbols, (A, :+))
            return :( $gA = LoopVectorization.SIMDPirates.vadd($gA, $B ))
        elseif (@capture(x, A_ -= B_) || @capture(x, A_ = A_ - B_)) && A isa Symbol
            # @show A, typeof(A)
            gA = get!(() -> gensym(A), reduction_symbols, (A, :-))
            return :( $gA = LoopVectorization.SIMDPirates.vadd($gA, $B ))
        elseif (@capture(x, A_ *= B_) || @capture(x, A_ = A_ * B_) || @capture(x, A_ = B_ * A_)) && A isa Symbol
            # @show A, typeof(A)
            gA = get!(() -> gensym(A), reduction_symbols, (A, :*))
            return :( $gA = LoopVectorization.SIMDPirates.vmul($gA, $B ))
        elseif (@capture(x, A_ /= B_) || @capture(x, A_ = A_ / B_)) && A isa Symbol
            # @show A, typeof(A)
            gA = get!(() -> gensym(A), reduction_symbols, (A, :/))
            return :( $gA = LoopVectorization.SIMDPirates.vmul($gA, $B ))
        elseif @capture(x, A_[i_])
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
                load_expr = :(LoopVectorization.SIMDPirates.vload($V, $pA + $itersym ))
                # load_expr = :(LoopVectorization.SIMDPirates.vload($V, $pA, $itersym))
            elseif isa(i, Expr)
                contains_itersym, i2 = subsymbol(i, declared_iter_sym, itersym)
                if contains_itersym
                    load_expr = :(LoopVectorization.SIMDPirates.vload($V, $pA + $i2 ))
                else
                    load_expr = :(LoopVectorization.SIMDPirates.vbroadcast($V, unsafe_load($pA, $i)))
                end
            else
                load_expr = :(LoopVectorization.SIMDPirates.vbroadcast($V, unsafe_load($pA, $i)))
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
        elseif @capture(x, A_[i_, j_])
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
                stridexpr = :(LoopVectorization.stride_row($A))
                if stridexpr ∈ keys(loop_constants_dict)
                    stridesym = loop_constants_dict[stridexpr]
                else
                    stridesym = gensym(:stride)
                    push!(loop_constants_quote.args, :( $stridesym = $stridexpr ))
                    loop_constants_dict[stridexpr] = stridesym
                end
                load_expr = :(LoopVectorization.SIMDPirates.vload($V, $pA + $itersym + $ej*$stridesym))
            elseif j == declared_iter_sym
                throw("Indexing columns with vectorized loop variable is not supported.")
            else
                # when loading something not indexed by the loop variable,
                # we assume that the intension is to broadcast
                stridexpr = :(LoopVectorization.stride_row($A))
                if stridexpr ∈ keys(loop_constants_dict)
                    stridesym = loop_constants_dict[stridexpr]
                else
                    stridesym = gensym(:stride)
                    push!(loop_constants_quote.args, :( $stridesym = $stridexpr ))
                    loop_constants_dict[stridexpr] = stridesym
                end
                load_expr = :(LoopVectorization.SIMDPirates.vbroadcast($V, unsafe_load($pA + $i + $ej*$stridesym)))
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
                # pA = esc(gensym(A))
                # pA = esc(Symbol(:p,A))
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

            stridexpr = :(LoopVectorization.stride_row($A))
            if stridexpr ∈ keys(loop_constants_dict)
                stridesym = loop_constants_dict[stridexpr]
            else
                stridesym = gensym(:stride)
                push!(loop_constants_quote.args, :( $stridesym = $stridexpr ))
                loop_constants_dict[stridexpr] = stridesym
            end
            # numiterexpr = :(LoopVectorization.num_row_strides($A))
            # if numiterexpr ∈ keys(loop_constants_dict)
            #     numitersym = loop_constants_dict[numiterexpr]
            # else
            #     numitersym = gensym(:numiter)
            #     push!(loop_constants_quote.args, :( $numitersym = $numiterexpr ))
            #     loop_constants_dict[numiterexpr] = numitersym
            # end

            expr = quote
                $br = LoopVectorization.extract_data.($B)

                # for $coliter ∈ 0:$numitersym-1
                for $coliter ∈ 0:length($br)-1
                    @inbounds LoopVectorization.SIMDPirates.vstore!($pA + $isym + $stridesym * $coliter, getindex($br,1+$coliter))
                end
            end

            return expr
        # elseif @capture(x, @nexprs N_ ex_)
        #     # println("Macroexpanding x:", x)
        #     # @show ex
        #     # mx = Expr(:escape, Expr(:block, Any[ Base.Cartesian.inlineanonymous(ex,i) for i = 1:N ]...))
        #     mx = Expr(:block, Any[ Base.Cartesian.inlineanonymous(ex,i) for i = 1:N ]...)
        #     # println("Macroexpanded x:", mx)
        #     return mx
        elseif @capture(x, zero(T_))
            return :(zero($V))
        elseif @capture(x, one(T_))
            return :(one($V))
        elseif @capture(x, B_ ? A_ : C_)
            return :(LoopVectorization.SIMDPirates.vifelse($B, $A, $C))
        elseif x == declared_iter_sym
            isymvec = gensym(itersym)
            push!(pre_quote.args, :($isymvec = SVec($(Expr(:tuple, [:(Core.VecElement{$VET}($(w-W))) for w ∈ 1:W]...)))))
            push!(main_body.args, :($isymvec = SIMDPirates.vadd($isymvec, vbroadcast($V, $W)) ))
            return isymvec
        else
            # println("Returning x:", x)
            return x
        end
    end, VectorizationDict, false) # macro_escape = false
end

"""
subsymbol(expr, i, j)
substitute symbol i with symbol j in expr
Returns true if a substitution was made, false otherwise.
"""
function subsymbol(expr, i, j)
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
macro vectorize(expr)
    if @capture(expr, for n_ ∈ 1:N_ body__ end)
        # q = vectorize_body(N, Float64, n, body, false)
        q = vectorize_body(N, Float64, 1, n, body)
    # elseif @capture(expr, for n_ ∈ 1:N_ body__ end)
    #     q = vectorize_body(N, element_type(body)
    elseif @capture(expr, for n_ ∈ eachindex(A_) body__ end)
        q = vectorize_body(:(length($A)), Float64, 1, n, body)
    elseif @capture(expr, for n_ ∈ eachindex(args__) body__ end)
        q = vectorize_body(:(min($([:(length($a)) for a ∈ args]...))), Float64, 1, n, body)
    else
        throw("Could not match loop expression.")
    end
    esc(q)
end
macro vectorize(type::Union{Symbol,DataType}, expr)
    if @capture(expr, for n_ ∈ 1:N_ body__ end)
        # q = vectorize_body(N, type, n, body, true)
        q = vectorize_body(N, type, 1, n, body)
    elseif @capture(expr, for n_ ∈ eachindex(A_) body__ end)
        q = vectorize_body(:(length($A)), type, 1, n, body)
    elseif @capture(expr, for n_ ∈ eachindex(args__) body__ end)
        q = vectorize_body(:(min($([:(length($a)) for a ∈ args]...))), type, 1, n, body)
    else
        throw("Could not match loop expression.")
    end
    esc(q)
end
macro vectorize(unroll_factor::Integer, expr)
    if @capture(expr, for n_ ∈ 1:N_ body__ end)
        # q = vectorize_body(N, type, n, body, true)
        q = vectorize_body(N, Float64, unroll_factor, n, body)
    elseif @capture(expr, for n_ ∈ eachindex(A_) body__ end)
        q = vectorize_body(:(length($A)), Float64, unroll_factor, n, body)
    elseif @capture(expr, for n_ ∈ eachindex(args__) body__ end)
        q = vectorize_body(:(min($([:(length($a)) for a ∈ args]...))), Float64, unroll_factor, n, body)
    else
        throw("Could not match loop expression.")
    end
    esc(q)
end
macro vectorize(type, unroll_factor::Integer, expr)
    if @capture(expr, for n_ ∈ 1:N_ body__ end)
        # q = vectorize_body(N, type, n, body, true)
        q = vectorize_body(N, type, unroll_factor, n, body)
    elseif @capture(expr, for n_ ∈ eachindex(A_) body__ end)
        q = vectorize_body(:(length($A)), type, unroll_factor, n, body)
    elseif @capture(expr, for n_ ∈ eachindex(args__) body__ end)
        q = vectorize_body(:(min($([:(length($a)) for a ∈ args]...))), type, unroll_factor, n, body)
    else
        throw("Could not match loop expression.")
    end
    esc(q)
end


end # module
