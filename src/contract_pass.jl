


contract_pass(x) = x # x will probably be a symbol
function contract_pass(expr::Expr)::Expr
    prewalk(expr) do ex
        if !(ex isa Expr)
            return ex
        elseif ex.head != :call
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
            elseif ex.head != :call
                ex
            end
        elseif @capture(ex, f_(c_, g_(a_, b_))) || @capture(ex, f_(g_(a_,b_), c_))
            if (f === :(+) || f == :(Base.FastMath.add_fast)) && (g === :(*) || g == :(Base.FastMath.mul_fast)) 
                if a isa Expr && a.head === :call && (first(a.args) === :(-) || first(a.args) == :(Base.FastMath.sub_fast))
                    Expr(:call, :vnfmadd, a, b, c)
                else
                    Expr(:call, :vmuladd, a, b, c) #Expr(:call, :vfmadd, a, b, c)
                end
            elseif (f === :(-) || f == :(Base.FastMath.sub_fast)) && (g === :(*) || g == :(Base.FastMath.mul_fast)) 
                if a isa Expr && a.head === :call && (first(a.args) === :(-) || first(a.args) == :(Base.FastMath.sub_fast))
                    Expr(:call, :vnfmsub, a, b, c)
                else
                    Expr(:call, :vfmsub, a, b, c)
                end
            else
                ex
            end
        else
            ex
        end
    end
end



