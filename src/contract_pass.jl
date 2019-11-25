


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


using MLStyle
walk(x, inner, outer) = outer(x)
walk(x::Expr, inner, outer) = outer(Expr(x.head, map(inner, x.args)...))

"""
    postwalk(f, expr)
Applies `f` to each node in the given expression tree, returning the result.
`f` sees expressions *after* they have been transformed by the walk. See also
`prewalk`.
"""
postwalk(f, x) = walk(x, x -> postwalk(f, x), f)

"""
    prewalk(f, expr)
Applies `f` to each node in the given expression tree, returning the result.
`f` sees expressions *before* they have been transformed by the walk, and the
walk will be applied to whatever `f` returns.
This makes `prewalk` somewhat prone to infinite loops; you probably want to try
`postwalk` first.
"""
prewalk(f, x)  = walk(f(x), x -> prewalk(f, x), identity)


function contract(expr)
    @match expr begin
        quote
            $a * $b + $c
end

function contract_pass(expr)
    prewalk(expr) do ex

    end
    @match expr begin
           quote
               struct $name{$tvar}
                   $f1 :: $t1
                   $f2 :: $t2
               end
           end =>
           quote
               struct $name{$tvar}
                   $f1 :: $t2
                   $f2 :: $t1
               end
           end |> rmlines
       end


end


