module LoopVectorization

using VectorizationBase, SIMDPirates, SLEEF, MacroTools
using VectorizationBase: REGISTER_SIZE, extract_data
using SIMDPirates: VECTOR_SYMBOLS
using MacroTools: @capture, prewalk, postwalk

export vectorizable, @vectorize



const SLEEFDict = Dict{Symbol,Expr}(
    :sin => :(SLEEF.sin_fast),
    :sinpi => :(SLEEF.sinpi),
    :cos => :(SLEEF.cos_fast),
    :cospi => :(SLEEF.cospi),
    :tan => :(SLEEF.tan_fast),
    :log => :(SLEEF.log_fast),
    :log10 => :(SLEEF.log10),
    :log2 => :(SLEEF.log2),
    :log1p => :(SLEEF.log1p),
    :exp => :(SLEEF.exp),
    :exp2 => :(SLEEF.exp2),
    :exp10 => :(SLEEF.exp10),
    :expm1 => :(SLEEF.expm1),
    :sqrt => :(SLEEF.sqrt), # faster than sqrt_fast
    :rsqrt => :(LoopVectorization.SIMDPirates.rsqrt),
    :cbrt => :(SLEEF.cbrt_fast),
    :asin => :(SLEEF.asin_fast),
    :acos => :(SLEEF.acos_fast),
    :atan => :(SLEEF.atan_fast),
    :sinh => :(SLEEF.sinh),
    :cosh => :(SLEEF.cosh),
    :tanh => :(SLEEF.tanh),
    :asinh => :(SLEEF.asinh),
    :acosh => :(SLEEF.acosh),
    :atanh => :(SLEEF.atanh),
    # :erf => :(SLEEF.erf),
    # :erfc => :(SLEEF.erfc),
    # :gamma => :(SLEEF.gamma),
    # :lgamma => :(SLEEF.lgamma),
    :trunc => :(SLEEF.trunc),
    :floor => :(SLEEF.floor),
    :ceil => :(SLEEF.ceil),
    :abs => :(SLEEF.abs),
    :sincos => :(SLEEF.sincos_fast),
    # :sincospi => :(SLEEF.sincospi_fast),
    # :pow => :(SLEEF.pow),
    # :hypot => :(SLEEF.hypot_fast),
    :mod => :(SLEEF.mod)
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


# """
# N is length of the vectors.
# T is the type of the index.
# n is the index.
# body is the body of the function.
# """
# function vectorize_body(N::Integer, T::DataType, unroll_factor, n, body, VectorizationDict = SLEEFDict)
#     T_size = sizeof(T)
#     W = REGISTER_SIZE ÷ T_size
#     while W > 2N
#         W >>= 1
#     end
#     WT = W * T_size
#     Q, r = divrem(N, W) #Assuming Mₖ is a multiple of W
#     QQ, Qr = divrem(Q, unroll_factor)
#     if r > 0
#         if unroll_factor == 1
#             QQ += 1
#         else
#             Qr += 1
#         end
#         Q += 1
#     end
#     # unroll the remainder iteration
#     # so that whenever Q >= unroll_factor, we will always have at least
#     # unroll_factor operations scheduled at a time.
#     if QQ > 0 && Qr > 0 && Qr < unroll_factor # if r > 0, Qr may equal 4
#         QQ -= 1
#         Qr += unroll_factor
#     end
#     V = Vec{W,T}
#
#
#     # body = _pirate(body)
#
#     # indexed_expressions = Dict{Symbol,Expr}()
#     indexed_expressions = Dict{Symbol,Symbol}() # Symbol, gensymbol
#     reduction_expressions = Dict{Symbol,Symbol}() # ParamSymbol,
#     # itersym = esc(gensym(:iter))
#     # itersym = esc(:iter)
#     # itersym = gensym(:iter)
#     isym = gensym(:i)
#     itersym = isym
#     # walk the expression, searching for all get index patterns.
#     # these will be replaced with
#     # Plan: definition of q will create vectorizables
#     main_body = quote end
#     reduction_symbols = Symbol[]
#     loaded_exprs = Dict{Expr,Symbol}()
#     loop_constants_dict = Dict{Expr,Symbol}()
#     loop_constants_quote = quote end
#
#     for b ∈ body
#         ## body preamble must define indexed symbols
#         ## we only need that for loads.
#         push!(main_body.args,
#             _vectorloads!(main_body, indexed_expressions, reduction_expressions, reduction_symbols, loaded_exprs, V, loop_constants_quote, loop_constants_dict, b;
#                             itersym = itersym, declared_iter_sym = n)
#         )# |> x -> (@show(x), _pirate(x)))
#     end
#
#     ### now we walk the body to look for reductions
#     if length(reduction_symbols) > 0
#         reductions = true
#     else
#         reductions = false
#     end
#
#     # q = quote end
#     q = loop_constants_quote
#     for (sym, psym) ∈ indexed_expressions
#         push!(q.args, :( $psym = vectorizable($sym) ))
#     end
#
#
#     # @show QQ, Qr, Q, r
#     # loop_body = [:($itersym = $isym), main_body]
#     loop_body = [main_body]
#     for unroll ∈ 1:unroll_factor-1
#         push!(loop_body, :($itersym = $isym + $(unroll*W)))
#         push!(loop_body, main_body)
#     end
#
#     if QQ > 0
#         push!(q.args,
#         quote
#             for $isym ∈ 1:$(unroll_factor*W):$(QQ*unroll_factor*W)
#                 $(loop_body...)
#             end
#         end)
#     end
#     for qri ∈ 1:Qr
#         push!(q.args,
#         quote
#             $itersym = $(QQ*unroll_factor*W + qri*W)
#             $main_body
#         end)
#     end
#     if r > 0
#         throw("Need to work on mask!")
#         maskuint = mask(r)
#         if W <= 8
#             mask = convert(UInt8, mask)
#         elseif W <= 16
#             mask = convert(UInt16, mask)
#         elseif W <= 32
#             mask = convert(UInt32, mask)
#         else
#             mask = maskuint
#         end
#         # mask = mask_expr(W, r)
#         iter = Q * W
#         r_body = quote end
#         for b ∈ body
#             push!(r_body.args, _spirate(prewalk(b) do x
#                 if @capture(x, A_[i_] = B_)
#                     if A ∉ keys(indexed_expressions)
#                         # pA = esc(gensym(A))
#                         # pA = esc(Symbol(:p,A))
#                         pA = Symbol(:p,A)
#                         indexed_expressions[A] = pA
#                     else
#                         pA = indexed_expressions[A]
#                     end
#                     if i == n
#                         return :(LoopVectorization.SIMDPirates.vstore($B, $pA, $iter, $mask))
#                     else
#                         return :(LoopVectorization.SIMDPirates.vstore($B, $pA, $i, $mask))
#                     end
#                 elseif @capture(x, A_[i_])
#                     if A ∉ keys(indexed_expressions)
#                         # pA = esc(gensym(A))
#                         # pA = esc(Symbol(:p,A))
#                         pA = Symbol(:p,A)
#                         indexed_expressions[A] = pA
#                     else
#                         pA = indexed_expressions[A]
#                     end
#                     if i == n
#                         return :(LoopVectorization.SIMDPirates.vload($V, $pA, $iter, $mask))
#                     else
#                         # when loading something not indexed by the loop variable,
#                         # we assume that the intension is to broadcast
#                         return :(LoopVectorization.SIMDPirates.vbroadcast($V, unsafe_load($pA, $i-1)))
#                     end
#                 else
#                     return x
#                 end
#             end, VectorizationDict, false)) # macro_escape = false
#         end
#         push!(q.args, r_body)
#     end
#     q
# end
function vectorize_body(N, Tsym::Symbol, uf, n, body, vecdict = SLEEFDict, VType = SVec)
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
function vectorize_body(N::Union{Symbol, Expr}, T::DataType, unroll_factor, n, body, vecdict = SLEEFDict, VType = SVec)
    unroll_factor == 1 || throw("Only unroll factor of 1 is currently supported. Was set to $unroll_factor.")
    T_size = sizeof(T)
    W = REGISTER_SIZE ÷ T_size
    # @show W, REGISTER_SIZE, T_size
    # @show T
    WT = W * T_size
    V = VType{W,T}

    # @show body

    # body = _pirate(body)

    # indexed_expressions = Dict{Symbol,Expr}()
    indexed_expressions = Dict{Symbol,Symbol}() # Symbol, gensymbol
    reduction_expressions = Dict{Symbol,Symbol}() # ParamSymbol,
    # itersym = esc(gensym(:iter))
    # itersym = esc(:iter)
    # itersym = :iter
    # itersym = gensym(:iter)
    isym = gensym(:i)
    itersym = isym
    # walk the expression, searching for all get index patterns.
    # these will be replaced with
    # Plan: definition of q will create vectorizables
    Nsym = gensym(:N)
    main_body = quote end
    reduction_symbols = Dict{Tuple{Symbol,Symbol},Symbol}()
    loaded_exprs = Dict{Expr,Symbol}()
    loop_constants_dict = Dict{Expr,Symbol}()
    loop_constants_quote = quote end

    for b ∈ body
        b = nexprs_expansion(b)
        ## body preamble must define indexed symbols
        ## we only need that for loads.
        push!(main_body.args,
            _vectorloads!(main_body, indexed_expressions, reduction_expressions, reduction_symbols, loaded_exprs, V, loop_constants_quote, loop_constants_dict, b;
                            itersym = itersym, declared_iter_sym = n, VectorizationDict = vecdict)
        )# |> x -> (@show(x), _pirate(x)))
    end
    # @show main_body


    # q = quote
    #     # QQ, Qr = divrem(Q, $unroll_factor)
    #     # if r > 0
    #     #     # $(unroll_factor == 1 ? :QQ : :Qr) += 1
    #     #     Qr += 1
    #     #     # Q += 1
    #     # end
    # end
    # pushfirst!(q.args, :((Q, r) = $(num_vector_load_expr(:LoopVectorization, N, W))))
    Qsym = gensym(:Q)
    remsym = gensym(:rem)
    q = quote
        $Nsym = $N
        ($Qsym, $remsym) = $(num_vector_load_expr(:LoopVectorization, N, W))
        # $loop_constants_quote
    end
    for (sym, psym) ∈ indexed_expressions
        push!(q.args, :( $psym = vectorizable($sym) ))
    end
    push!(q.args, loop_constants_quote)
    # @show QQ, Qr, Q, r
    # loop_body = [:($itersym = $isym), :($main_body)]
    loop_body = [main_body]
    for unroll ∈ 1:unroll_factor-1
        push!(loop_body, :($itersym = $isym + $(unroll*W)))
        push!(loop_body, :($main_body))
    end
    push!(q.args,
    quote
        for $isym ∈ 0:$(unroll_factor*W):($Qsym*$(unroll_factor*W)-1)
            $(loop_body...)
        end
    end)
    # if unroll_factor > 1
    #     push!(q.args,
    #     quote
    #         for $isym ∈ 1:Qr
    #             $itersym = QQ*$(unroll_factor*WT) + $isym*$WT
    #             $main_body
    #         end
    #     end)
    # end
    Itype = Base.Threads.inttype(T)
    masksym = gensym(:mask)
    masked_loop_body = add_masks.(loop_body, masksym)
    push!(q.args, quote
        if $remsym > 0
            $masksym = $(mask_expr(W, remsym))
            $isym = ($Nsym - $remsym)
            $(masked_loop_body...)
        end
    end)


    ### now we walk the body to look for reductions
    for ((sym,op),gsym) ∈ reduction_symbols
        if op == :+ || op == :-
            pushfirst!(q.args, :($gsym = LoopVectorization.SIMDPirates.vbroadcast($V,zero($T))))
        elseif op == :* || op == :/
            pushfirst!(q.args, :($gsym = LoopVectorization.SIMDPirates.vbroadcast($V,one($T))))
        end
        if op == :+
            push!(q.args, :(@fastmath $sym = $sym + LoopVectorization.SIMDPirates.vsum($gsym)))
        elseif op == :-
            push!(q.args, :(@fastmath $sym = $sym - LoopVectorization.SIMDPirates.vsum($gsym)))
        elseif op == :*
            push!(q.args, :(@fastmath $sym = $sym * LoopVectorization.SIMDPirates.vprod($gsym)))
        elseif op == :/
            push!(q.args, :(@fastmath $sym = $sym / LoopVectorization.SIMDPirates.vprod($gsym)))
        end
    end

    q
end
function add_masks(expr, masksym)
    postwalk(expr) do x
        if @capture(x, LoopVectorization.SIMDPirates.vstore(V_, ptr_))
            return :(LoopVectorization.SIMDPirates.vstore($V, $ptr, $masksym))
        elseif @capture(x, LoopVectorization.SIMDPirates.vload(V_, ptr_))
            return :(LoopVectorization.SIMDPirates.vload($V, $ptr, $masksym))
        else
            return x
        end
    end
end

function _vectorloads(V, expr; itersym = :iter, declared_iter_sym = nothing, VectorizationDict = SLEEFDict)


    # body = _pirate(body)

    # indexed_expressions = Dict{Symbol,Expr}()
    indexed_expressions = Dict{Symbol,Symbol}() # Symbol, gensymbol
    reduction_expressions = Dict{Symbol,Symbol}() # ParamSymbol,

    main_body = quote end
    reduction_symbols = Dict{Tuple{Symbol,Symbol},Symbol}()
    loaded_exprs = Dict{Expr,Symbol}()
    loop_constants_dict = Dict{Expr,Symbol}()
    loop_constants_quote = quote end

    push!(main_body.args,
        _vectorloads!(main_body, indexed_expressions, reduction_expressions, reduction_symbols, loaded_exprs, V, loop_constants_quote, loop_constants_dict, expr;
            itersym = itersym, declared_iter_sym = declared_iter_sym, VectorizationDict = VectorizationDict)
    )
    main_body
end

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

function _vectorloads!(main_body, indexed_expressions, reduction_expressions, reduction_symbols, loaded_exprs, V, loop_constants_quote, loop_constants_dict, expr;
                            itersym = :iter, declared_iter_sym = nothing, VectorizationDict = SLEEFDict)
    _spirate(prewalk(expr) do x
        # @show x
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
                return :(LoopVectorization.SIMDPirates.vstore($B, $pA + $itersym))
            else
                return :(LoopVectorization.SIMDPirates.vstore($B, $pA + $i))
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
                return :(LoopVectorization.SIMDPirates.vstore($B, $pA + $itersym + $ej*$stridesym ))
            else
                throw("Indexing columns with vectorized loop variable is not supported.")
            end
        elseif @capture(x, A_ += B_) || @capture(x, A_ = A_ + B_) || @capture(x, A_ = B_ + A_)
            gA = get!(() -> gensym(A), reduction_symbols, (A, :+))
            return :( $gA = $gA + $B )
        elseif @capture(x, A_ -= B_) || @capture(x, A_ = A_ - B_)
            gA = get!(() -> gensym(A), reduction_symbols, (A, :-))
            return :( $gA = $gA + $B )
        elseif @capture(x, A_ *= B_) || @capture(x, A_ = A_ * B_) || @capture(x, A_ = B_ * A_)
            gA = get!(() -> gensym(A), reduction_symbols, (A, :*))
            return :( $gA = LoopVectorization.SIMDPirates.evmul($gA, $B ))
        elseif @capture(x, A_ /= B_) || @capture(x, A_ = A_ / B_)
            gA = get!(() -> gensym(A), reduction_symbols, (A, :/))
            return :( $gA = LoopVectorization.SIMDPirates.evmul($gA, $B ))
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
            else
                load_expr = :(LoopVectorization.SIMDPirates.vbroadcast($V, unsafe_load($pA, $i)))
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
                    @inbounds LoopVectorization.SIMDPirates.vstore(getindex($br,1+$coliter), $pA + $isym + $stridesym * $coliter)
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
        else
            # println("Returning x:", x)
            return x
        end
    end, VectorizationDict, false) # macro_escape = false
end


"""
Arguments are
@vectorze Type UnrollFactor forloop

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
    else
        throw("Could not match loop expression.")
    end
    esc(q)
end
macro vectorize(type, unroll_factor, expr)
    if @capture(expr, for n_ ∈ 1:N_ body__ end)
        # q = vectorize_body(N, type, n, body, true)
        q = vectorize_body(N, type, unroll_factor, n, body)
    elseif @capture(expr, for n_ ∈ eachindex(A_) body__ end)
        q = vectorize_body(:(length($A)), type, unroll_factor, n, body)
    else
        throw("Could not match loop expression.")
    end
    esc(q)
end


end # module
