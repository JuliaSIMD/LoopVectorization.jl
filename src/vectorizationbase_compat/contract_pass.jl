


function check_negative(x)
    x isa Expr || return false
    x.head === :call || return false
    length(x.args) == 2 || return false
    a = first(x.args)
    return (a === :(-) || a == :(Base.FastMath.sub_fast))
end


mulexprcost(::Number) = 0
mulexprcost(::Symbol) = 1
function mulexprcost(ex::Expr)
    base = ex.head === :call ? 10 : 1
    base + length(ex.args)
end
function mulexpr(mulexargs)
    a = (mulexargs[1])::Union{Symbol,Expr,Number}
    if length(mulexargs) == 2
        return (a, mulexargs[2]::Union{Symbol,Expr,Number})
    elseif length(mulexargs) == 3
        # We'll calc the product between the guesstimated cheaper two args first, for better out of order execution
        b = (mulexargs[2])::Union{Symbol,Expr,Number}
        c = (mulexargs[3])::Union{Symbol,Expr,Number}
        ac = mulexprcost(a)
        bc = mulexprcost(b)
        cc = mulexprcost(c)
        maxc = max(ac, bc, cc)
        if ac == maxc
            return (a, Expr(:call, :vmul, b, c))
        elseif bc == maxc
            return (b, Expr(:call, :vmul, a, c))
        else
            return (c, Expr(:call, :vmul, a, b))
        end
    else
        return (a, Expr(:call, :vmul, @view(mulexargs[2:end])...)::Expr)
    end
    a = (mulexargs[1])::Union{Symbol,Expr,Number}
    b = if length(mulexargs) == 2 # two arg mul
        (mulexargs[2])::Union{Symbol,Expr,Number}
    else
        Expr(:call, :vmul, @view(mulexargs[2:end])...)::Expr
    end
    a, b
end
function append_args_skip!(call, args, i)
    for j ∈ eachindex(args)
        j == i && continue
        push!(call.args, args[j])
    end
    call
end



function recursive_muladd_search!(call, argv, cnmul::Bool = false, csub::Bool = false)
    length(argv) < 3 && return length(call.args) == 4, cnmul, csub
    fun = first(argv)
    isadd = fun === :+ || fun === :vadd! || fun === :vadd || fun == :(Base.FastMath.add_fast)
    issub = fun === :- || fun === :vsub! || fun === :vsub || fun == :(Base.FastMath.sub_fast)
    if !(isadd | issub)
        return length(call.args) == 4, cnmul, csub
    end
    exargs = @view(argv[2:end])
    issub && @assert length(exargs) == 2
    for (i,ex) ∈ enumerate(exargs)
        if ex isa Expr && ex.head === :call
            exa = ex.args
            f = first(exa)
            exav = @view(exa[2:end])
            if f === :* || f === :vmul! || f === :vmul || f == :(Base.FastMath.mul_fast)
                # isnmul = any(check_negative, exav)
                a, b = mulexpr(exav)
                call.args[2] = a
                call.args[3] = b
                if length(exargs) == 2
                    push!(call.args, exargs[3 -  i])
                else
                    push!(call.args, append_args_skip!(Expr(:call, :+), exargs, i))
                end
                if issub
                    csub = i == 1
                    cnmul = !csub
                end
                return true, cnmul, csub
            elseif isadd
                found, cnmul, csub = recursive_muladd_search!(call, exa)
                if found
                    if csub
                        call.args[4] = if length(exargs) == 2
                            Expr(:call, :-, exargs[3 - i], call.args[4])
                        else
                            Expr(:call, :-, append_args_skip!(Expr(:call, :+), exargs, i), call.args[4])
                        end
                    else
                        call.args[4] = append_args_skip!(Expr(:call, :+, call.args[4]), exargs, i)
                    end
                    return true, cnmul, false
                end
            elseif issub
                found, cnmul, csub = recursive_muladd_search!(call, exa)
                if found
                    if i == 1
                        if csub
                            call.args[4] = Expr(:call, :+, call.args[4], exargs[3 - i])
                        else
                            call.args[4] = Expr(:call, :-, call.args[4], exargs[3 - i])
                        end
                    else
                        cnmul = !cnmul
                        if csub
                            call.args[4] = Expr(:call, :+, exargs[3 - i], call.args[4])
                        else
                            call.args[4] = Expr(:call, :-, exargs[3 - i], call.args[4])
                        end
                        csub = false
                    end
                    return true, cnmul, csub
                end                
            end
        end
    end
    length(call.args) == 4, cnmul, csub
end
                          
function capture_muladd(ex::Expr, mod, LHS = nothing)
    call = Expr(:call, Symbol(""), Symbol(""), Symbol(""))
    found, nmul, sub = recursive_muladd_search!(call, ex.args)
    found || return ex
    # a, b, c = call.args[2], call.args[3], call.args[4]
    # call.args[2], call.args[3], call.args[4] = c, a, b
    clobber = false#call.args[4] == LHS
    f = if nmul && sub
        clobber ? :vfnmsub231 : :vfnmsub_fast
    elseif nmul
        clobber ? :vfnmadd231 : :vfnmadd_fast
    elseif sub
        clobber ? :vfmsub231 : :vfmsub_fast
    else
        clobber ? :vfmadd231 : :vfmadd_fast
    end
    if mod === nothing
        call.args[1] = f
    else
        call.args[1] = Expr(:(.), mod, QuoteNote(f))#_fast))
    end
    call
end

contract_pass!(::Any, ::Any) = nothing
function contract!(expr::Expr, ex::Expr, i::Int, mod = nothing)
    # if ex.head === :call
        # expr.args[i] = capture_muladd(ex, mod)
    if ex.head === :(+=)
        call = Expr(:call, :vadd)
        append!(call.args, ex.args)
        expr.args[i] = ex = Expr(:(=), first(ex.args), call)
    elseif ex.head === :(-=)
        call = Expr(:call, :vsub)
        append!(call.args, ex.args)
        expr.args[i] = ex = Expr(:(=), first(ex.args), call)
    elseif ex.head === :(*=)
        call = Expr(:call, :vmul)
        append!(call.args, ex.args)
        expr.args[i] = ex = Expr(:(=), first(ex.args), call)
    elseif ex.head === :(/=)
        call = Expr(:call, :vfdiv)
        append!(call.args, ex.args)
        expr.args[i] = ex = Expr(:(=), first(ex.args), call)
    end
    if ex.head === :(=)
        RHS = ex.args[2]
        # @show ex
        if RHS isa Expr && RHS.head === :call
            ex.args[2] = capture_muladd(RHS, mod, ex.args[1])
        end
    end
    contract_pass!(expr.args[i], mod)
end
# contract_pass(x) = x # x will probably be a symbol
function contract_pass!(expr::Expr, mod = nothing)
    for (i,ex) ∈ enumerate(expr.args)
        ex isa Expr || continue
        contract!(expr, ex, i, mod)
    end
end

