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
    :rsqrt => :(SIMDPirates.rsqrt),
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
    :pow => :(SLEEF.pow),
    # :hypot => :(SLEEF.hypot_fast),
    :mod => :(SLEEF.mod)
    # :copysign => :copysign
)





function _spirate(ex, dict, macro_escape = true)
    ex = postwalk(ex) do x
        # @show x
        if @capture(x, SIMDPirates.vadd(SIMDPirates.vmul(a_, b_), c_)) || @capture(x, SIMDPirates.vadd(c_, SIMDPirates.vmul(a_, b_)))
            return :(SIMDPirates.vmuladd($a, $b, $c))
        elseif @capture(x, SIMDPirates.vadd(SIMDPirates.vmul(a_, b_), SIMDPirates.vmul(c_, d_), e_)) || @capture(x, SIMDPirates.vadd(SIMDPirates.vmul(a_, b_), e_, SIMDPirates.vmul(c_, d_))) || @capture(x, SIMDPirates.vadd(e_, SIMDPirates.vmul(a_, b_), SIMDPirates.vmul(c_, d_)))
            return :(SIMDPirates.vmuladd($a, $b, SIMDPirates.vmuladd($c, $d, $e)))
        elseif @capture(x, SIMDPirates.vadd(SIMDPirates.vmul(b_, c_), SIMDPirates.vmul(d_, e_), SIMDPirates.vmul(f_, g_), a_)) ||
                @capture(x, SIMDPirates.vadd(SIMDPirates.vmul(b_, c_), SIMDPirates.vmul(d_, e_), a_, SIMDPirates.vmul(f_, g_))) ||
                @capture(x, SIMDPirates.vadd(SIMDPirates.vmul(b_, c_), a_, SIMDPirates.vmul(d_, e_), SIMDPirates.vmul(f_, g_))) ||
                @capture(x, SIMDPirates.vadd(a_, SIMDPirates.vmul(b_, c_), SIMDPirates.vmul(d_, e_), SIMDPirates.vmul(f_, g_)))
            return :(SIMDPirates.vmuladd($g, $f, SIMDPirates.vmuladd($e, $d, SIMDPirates.vmuladd($c, $b, $a))))
        elseif @capture(x, a_ * b_ + c_ - c_) || @capture(x, c_ + a_ * b_ - c_) || @capture(x, a_ * b_ - c_ + c_) || @capture(x, - c_ + a_ * b_ + c_)
            return :(SIMDPirates.vmul($a, $b))
        elseif @capture(x, a_ * b_ + c_ - d_) || @capture(x, c_ + a_ * b_ - d_) || @capture(x, a_ * b_ - d_ + c_) || @capture(x, - d_ + a_ * b_ + c_) || @capture(x, SIMDPirates.vsub(SIMDPirates.vmuladd(a_, b_, c_), d_))
            return :(SIMDPirates.vmuladd($a, $b, SIMDPirates.vsub($c, $d)))
        elseif @capture(x, a_ += b_)
            return :($a = SIMDPirates.vadd($a, $b))
        elseif @capture(x, a_ -= b_)
            return :($a = SIMDPirates.vsub($a, $b))
        elseif @capture(x, a_ *= b_)
            return :($a = SIMDPirates.vmul($a, $b))
        elseif @capture(x, a_ /= b_)
            return :($a = SIMDPirates.vdiv($a, $b))
        elseif @capture(x, a_ / sqrt(b_))
            return :($a * rsqrt($b))
        elseif @capture(x, inv(sqrt(a_)))
            return :(rsqrt($a))
        elseif @capture(x, @horner a__)
            return horner(a...)
        elseif @capture(x, Base.Math.muladd(a_, b_, c_))
            return :( SIMDPirates.vmuladd($a, $b, $c) )
        elseif isa(x, Symbol) && !occursin("@", string(x))
            vec_sym = get(dict, x, :not_found)
            if vec_sym != :not_found
                return vec_sym
            else
                vec_sym = get(VECTOR_SYMBOLS, x, :not_found)
                return vec_sym == :not_found ? x : :(SIMDPirates.$(vec_sym))
            end
        else
            return x
        end
    end
    macro_escape ? esc(ex) : ex
end







mask_expr(W, r) = :($(Expr(:tuple, [i > r ? Core.VecElement{Bool}(false) : Core.VecElement{Bool}(true) for i ∈ 1:W]...)))

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

"""
N is length of the vectors.
T is the type of the index.
n is the index.
body is the body of the function.
"""
function vectorize_body(N::Integer, T::DataType, unroll_factor, n, body, VectorizationDict = SLEEFDict)
    T_size = sizeof(T)
    W = REGISTER_SIZE ÷ T_size
    while W > 2N
        W >>= 1
    end
    WT = W * T_size
    Q, r = divrem(N, W) #Assuming Mₖ is a multiple of W
    QQ, Qr = divrem(Q, unroll_factor)
    if r > 0
        if unroll_factor == 1
            QQ += 1
        else
            Qr += 1
        end
        Q += 1
    end
    # unroll the remainder iteration
    # so that whenever Q >= unroll_factor, we will always have at least
    # unroll_factor operations scheduled at a time.
    if QQ > 0 && Qr > 0 && Qr < unroll_factor # if r > 0, Qr may equal 4
        QQ -= 1
        Qr += unroll_factor
    end
    V = SVec{W,T}


    # body = _pirate(body)

    # indexed_expressions = Dict{Symbol,Expr}()
    indexed_expressions = Dict{Symbol,Symbol}() # Symbol, gensymbol
    reduction_expressions = Dict{Symbol,Symbol}() # ParamSymbol,
    # itersym = esc(gensym(:iter))
    # itersym = esc(:iter)
    itersym = :iter
    isym = gensym(:i)
    # walk the expression, searching for all get index patterns.
    # these will be replaced with
    # Plan: definition of q will create vectorizables
    main_body = quote end
    reduction_symbols = Symbol[]
    loaded_exprs = Dict{Expr,Symbol}()

    for b ∈ body
        ## body preamble must define indexed symbols
        ## we only need that for loads.
        push!(main_body.args,
            _vectorloads!(main_body, indexed_expressions, reduction_expressions, reduction_symbols, loaded_exprs, V, b;
                            itersym = itersym, declared_iter_sym = n)
        )# |> x -> (@show(x), _pirate(x)))
    end

    ### now we walk the body to look for reductions
    if length(reduction_symbols) > 0
        reductions = true
    else
        reductions = false
    end

    q = quote end
    for (sym, psym) ∈ indexed_expressions
        push!(q.args, :( $psym = vectorizable($sym) ))
    end


    # @show QQ, Qr, Q, r
    loop_body = [:($itersym = $isym), main_body]
    for unroll ∈ 1:unroll_factor-1
        push!(loop_body, :($itersym = $isym + $(unroll*W)))
        push!(loop_body, main_body)
    end

    if QQ > 0
        push!(q.args,
        quote
            for $isym ∈ 1:$(unroll_factor*W):$(QQ*unroll_factor*W)
                $(loop_body...)
            end
        end)
    end
    for qri ∈ 1:Qr
        push!(q.args,
        quote
            $itersym = $(QQ*unroll_factor*W + qri*W)
            $main_body
        end)
    end
    if r > 0
        throw("Need to work on mask!")
        mask = mask_expr(W, r)
        iter = Q * W
        r_body = quote end
        for b ∈ body
            push!(r_body.args, _spirate(prewalk(b) do x
                if @capture(x, A_[i_] = B_)
                    if A ∉ keys(indexed_expressions)
                        # pA = esc(gensym(A))
                        # pA = esc(Symbol(:p,A))
                        pA = Symbol(:p,A)
                        indexed_expressions[A] = pA
                    else
                        pA = indexed_expressions[A]
                    end
                    if i == n
                        return :(SIMDPirates.vstore($B, $pA, $iter, $mask))
                    else
                        return :(SIMDPirates.vstore($B, $pA, $i, $mask))
                    end
                elseif @capture(x, A_[i_])
                    if A ∉ keys(indexed_expressions)
                        # pA = esc(gensym(A))
                        # pA = esc(Symbol(:p,A))
                        pA = Symbol(:p,A)
                        indexed_expressions[A] = pA
                    else
                        pA = indexed_expressions[A]
                    end
                    if i == n
                        return :(SIMDPirates.vload($V, $pA, $iter, $mask))
                    else
                        # when loading something not indexed by the loop variable,
                        # we assume that the intension is to broadcast
                        return :(SIMDPirates.vbroadcast($V, unsafe_load($pA, $i-1)))
                    end
                else
                    return x
                end
            end, VectorizationDict, false)) # macro_escape = false
        end
        push!(q.args, r_body)
    end
    q
end
function vectorize_body(N, Tsym::Symbol, uf, n, body)
    if Tsym == :Float32
        vectorize_body(N, Float32, uf, n, body)
    elseif Tsym == :Float64
        vectorize_body(N, Float64, uf, n, body)
    elseif Tsym == :ComplexF32
        vectorize_body(N, ComplexF32, uf, n, body)
    elseif Tsym == :ComplexF64
        vectorize_body(N, ComplexF64, uf, n, body)
    else
        throw("Type $Tsym is not supported.")
    end
end
function vectorize_body(N::Union{Symbol, Expr}, T::DataType, unroll_factor, n, body)
    unroll_factor == 1 || throw("Only unroll factor of 1 is currently supported. Was set to $unroll_factor.")
    T_size = sizeof(T)
    W = REGISTER_SIZE ÷ T_size
    # @show W, REGISTER_SIZE, T_size
    # @show T
    WT = W * T_size
    V = SVec{W,T}

    # @show body

    # body = _pirate(body)

    # indexed_expressions = Dict{Symbol,Expr}()
    indexed_expressions = Dict{Symbol,Symbol}() # Symbol, gensymbol
    reduction_expressions = Dict{Symbol,Symbol}() # ParamSymbol,
    # itersym = esc(gensym(:iter))
    # itersym = esc(:iter)
    # itersym = :iter
    itersym = gensym(:iter)
    isym = gensym(:i)
    # walk the expression, searching for all get index patterns.
    # these will be replaced with
    # Plan: definition of q will create vectorizables
    Nsym = gensym(:N)
    main_body = quote end
    reduction_symbols = Symbol[]
    loaded_exprs = Dict{Expr,Symbol}()

    for b ∈ body
        ## body preamble must define indexed symbols
        ## we only need that for loads.
        push!(main_body.args,
            _vectorloads!(main_body, indexed_expressions, reduction_expressions, reduction_symbols, loaded_exprs, V, b;
                            itersym = itersym, declared_iter_sym = n)
        )# |> x -> (@show(x), _pirate(x)))
    end
    # @show main_body

    ### now we walk the body to look for reductions
    if length(reduction_symbols) > 0
        reductions = true
    else
        reductions = false
    end

    # q = quote
    #     # QQ, Qr = divrem(Q, $unroll_factor)
    #     # if r > 0
    #     #     # $(unroll_factor == 1 ? :QQ : :Qr) += 1
    #     #     Qr += 1
    #     #     # Q += 1
    #     # end
    # end
    # pushfirst!(q.args, :((Q, r) = $(num_vector_load_expr(:LoopVectorization, N, W))))
    q = quote
        $Nsym = $N
        (Q, r) = $(num_vector_load_expr(:LoopVectorization, N, W))
    end
    for (sym, psym) ∈ indexed_expressions
        push!(q.args, :( $psym = vectorizable($sym) ))
    end
    # @show QQ, Qr, Q, r
    loop_body = [:($itersym = $isym), :($main_body)]
    for unroll ∈ 1:unroll_factor-1
        push!(loop_body, :($itersym = $isym + $(unroll*W)))
        push!(loop_body, :($main_body))
    end
    push!(q.args,
    quote
        for $isym ∈ 1:$(unroll_factor*W):(Q*$(unroll_factor*W))
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
    masked_loop_body = add_masks.(loop_body)
    push!(q.args, quote
        if r > 0
            mask = SIMDPirates.vless_or_equal(
                SIMDPirates.vsub(
                    $(Expr(:tuple, [:(Core.VecElement{$Itype}(($(Itype(w))))) for w ∈ 1:W]...) ),
                    unsafe_trunc($Itype, r) # unsafe trunc is safe unless CPU's vector width > 2^31. W just 2^4 for avx512 and Float32
                ), zero($Itype)
            )
            $isym = $Nsym - r + 1
            $(masked_loop_body...)
        end
    end)
    # push!(q.args,
    # quote
    #     for $n ∈ $N-r+1:$N
    #         $(body...)
    #     end
    # end)
    q
end
function add_masks(expr)
    postwalk(expr) do x
        if @capture(x, SIMDPirates.vstore(V_, ptr_, i_))
            return :(SIMDPirates.vstore($V, $ptr, $i, mask))
        elseif @capture(x, SIMDPirates.vload(V_, ptr_, i_))
            return :(SIMDPirates.vload($V, $ptr, $i, mask))
        else
            return x
        end
    end
end

function _vectorloads(V, expr; itersym = :iter, declared_iter_sym = nothing)


    # body = _pirate(body)

    # indexed_expressions = Dict{Symbol,Expr}()
    indexed_expressions = Dict{Symbol,Symbol}() # Symbol, gensymbol
    reduction_expressions = Dict{Symbol,Symbol}() # ParamSymbol,

    main_body = quote end
    reduction_symbols = Symbol[]
    loaded_exprs = Dict{Expr,Symbol}()

    push!(main_body.args,
        _vectorloads!(main_body, indexed_expressions, reduction_expressions, reduction_symbols, loaded_exprs, V, expr;
            itersym = itersym, declared_iter_sym = declared_iter_sym)
    )
    main_body
end

function _vectorloads!(main_body, indexed_expressions, reduction_expressions, reduction_symbols, loaded_exprs, V, expr;
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
                return :(SIMDPirates.vstore($B, $pA, $itersym))
            else
                return :(SIMDPirates.vstore($B, $pA, $i))
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
                return :(SIMDPirates.vstore($B, $pA, $itersym + $ej*LoopVectorization.stride_row($A)))
            else
                throw("Indexing columns with vectorized loop variable is not supported.")
            end
        elseif (@capture(x, A += B_) || @capture(x, A -= B_) || @capture(x, A *= B_) || @capture(x, A /= B_)) && A ∉ reduction_symbols
            push!(reduction_symbols, A)
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
                load_expr = :(SIMDPirates.vload($V, $pA, $itersym))
            else
                load_expr = :(SIMDPirates.vbroadcast($V, unsafe_load($pA, $i-1)))
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
                load_expr = :(SIMDPirates.vload($V, $pA, $itersym + $ej*LoopVectorization.stride_row($A)))
            elseif j == declared_iter_sym
                throw("Indexing columns with vectorized loop variable is not supported.")
            else
                # when loading something not indexed by the loop variable,
                # we assume that the intension is to broadcast
                load_expr = :(SIMDPirates.vbroadcast($V, unsafe_load($pA, $i + $ej*LoopVectorization.stride_row($A))))
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
            numiter = gensym(:numiter)
            stridesym = gensym(:stride)

            expr = quote
                $numiter = LoopVectorization.num_row_strides($A)
                $stridesym = LoopVectorization.stride_row($A)
                $br = LoopVectorization.extract_data.($B)

                for $coliter ∈ 0:$numiter-1
                    @inbounds SIMDPirates.vstore(getindex($br,1+$coliter), $pA, $isym + $stridesym * $coliter)
                end
            end

            return expr
        elseif @capture(x, @nexprs N_ ex_)
            # println("Macroexpanding x:", x)
            # @show ex
            # mx = Expr(:escape, Expr(:block, Any[ Base.Cartesian.inlineanonymous(ex,i) for i = 1:N ]...))
            mx = Expr(:block, Any[ Base.Cartesian.inlineanonymous(ex,i) for i = 1:N ]...)
            # println("Macroexpanded x:", mx)
            return mx
        elseif @capture(x, zero(T_))
            return :(zero($V))
        elseif @capture(x, one(T_))
            return :(one($V))
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
