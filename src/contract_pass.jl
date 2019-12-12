
function check_negative(x)
    x isa Expr || return false
    x.head === :call || return false
    length(x.args) == 2 || return false
    a = first(x.args)
    return (a === :(-) || a == :(Base.FastMath.sub_fast))
end

function capture_muladd(ex::Expr)
    # These are guaranteed by calling contract_pass
    # ex isa Expr || return ex
    # ex.head === :call || return ex
    args = ex.args
    f = first(args)::Union{Symbol,Expr}
    fplus = (f === :+)::Bool | (f == :(Base.FastMath.add_fast))
    fminus = (f === :-)::Bool | (f == :(Base.FastMath.sub_fast))
    (fplus | fminus) || return ex
    Nargs = length(args)
    Nargs > 2 || return ex
    j = 2
    while j ≤ Nargs
        argsⱼ = args[j]
        if argsⱼ isa Expr && (first(argsⱼ.args) === :* || first(argsⱼ.args) == :(Base.FastMath.mul_fast))
            break
        end
        j += 1
    end
    j > Nargs && return ex
    mulexpr::Expr = args[j]
    if Nargs == 3
        c = args[j == 2 ? 3 : 2]
    else
        c = Expr(:call, :vadd)
        for i ∈ 2:Nargs
            i == j || push!(c.args, args[i])
        end
    end
    isnmul = any(check_negative, @view(mulexpr.args[2:end]))
    a = mulexpr.args[2]
    b = if length(mulexpr.args) == 3 # two arg mul
        mulexpr.args[3]
    else
        Expr(:call, :vmul, @view(mulexpr.args[3:end])...)
    end
    cf = if fplus
        if isnmul
            :vfnmadd
        else
            :vmuladd
        end
    else
        if isnmul
            :vfnmsub
        else
            :vfnmadd
        end
    end
    Expr(:call, cf, a, b, c)
end


contract_pass(x) = x # x will probably be a symbol
function contract_pass(expr::Expr)::Expr
    prewalk(expr) do ex
        if !(ex isa Expr)
            return ex
        elseif ex.head !== :call
            if ex.head === :(+=)
                call = Expr(:call, :(+))
                append!(call.args, ex.args)
                Expr(:(=), first(ex.args), call)
            elseif ex.head === :(-=)
                call = Expr(:call, :(-))
                append!(call.args, ex.args)
                Expr(:(=), first(ex.args), call)
            elseif ex.head === :(*=)
                call = Expr(:call, :(*))
                append!(call.args, ex.args)
                Expr(:(=), first(ex.args), call)
            elseif ex.head === :(/=)
                call = Expr(:call, :(/))
                append!(call.args, ex.args)
                Expr(:(=), first(ex.args), call)
            else
                ex
            end
        else # ex.head === :call
            return capture_muladd(ex)
        end
    end
end

#         elseif @capture(ex, f_(c_, g_(a_, b_))) || @capture(ex, f_(g_(a_,b_), c_))
#             if (f === :(+) || f == :(Base.FastMath.add_fast)) && (g === :(*) || g == :(Base.FastMath.mul_fast)) 
#                 if a isa Expr && a.head === :call && (first(a.args) === :(-) || first(a.args) == :(Base.FastMath.sub_fast))
#                     Expr(:call, :vfnmadd, a, b, c)
#                 else
#                     Expr(:call, :vmuladd, a, b, c) #Expr(:call, :vfmadd, a, b, c)
#                 end
#             elseif (f === :(-) || f == :(Base.FastMath.sub_fast)) && (g === :(*) || g == :(Base.FastMath.mul_fast)) 
#                 if a isa Expr && a.head === :call && (first(a.args) === :(-) || first(a.args) == :(Base.FastMath.sub_fast))
#                     Expr(:call, :vfnmsub, a, b, c)
#                 else
#                     Expr(:call, :vfmsub, a, b, c)
#                 end
#             else
#                 ex
#             end
#         else
#             ex
#         end
#     end
# end



