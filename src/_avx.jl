using MacroTools: postwalk, prewalk
using LoopVectorization: LoopVectorization, LoopSet, lower

#----------------------------------------------------------------------------------------------------

struct Ex{T, Tup} end

function to_type(ex::Expr)
    Ex{ex.head, Tuple{to_type.(ex.args)...}}
end

to_type(x) = x
to_type(::LineNumberNode) = nothing

#----------------------------------------------------------------------------------------------------

to_expr(ex::Type{Ex{Head, Tup}}) where {Head, Tup} = Expr(Head, (to_expr(x) for x in Tup.parameters)...)
to_expr(x) = x

#----------------------------------------------------------------------------------------------------

function find_vars_and_gensym!(ex::Expr, vars::Dict{Symbol, Symbol}, ivars::Vector{Symbol})
    if ex.head == :(=) && ex.args[1] isa Symbol
        push!(ivars, ex.args[1])
    elseif ex.head == :call
        push!(ivars, ex.args[1])
    end
    ex
end

function find_vars_and_gensym!(x::Symbol, vars::Dict{Symbol, Symbol}, ivars::Vector{Symbol})
    if (x ∉ keys(vars)) && (x ∉ ivars)
        gx = gensym(x)
        push!(vars, x => gx)
        gx
    else
        get(vars, x, x)
    end    
end

find_vars_and_gensym!(x, vars::Dict{Symbol, Symbol}, ivars::Vector{Symbol}) = x

#----------------------------------------------------------------------------------------------------

nt(keys, vals) = NamedTuple{keys, typeof(vals)}(vals)

macro _avx(ex)
    D     = Dict{Symbol, Symbol}()
    ivars = Symbol[]

    gex = prewalk(x -> find_vars_and_gensym!(x, D, ivars), ex)

    type_ex = to_type(gex)

    tvars  = Tuple(keys(D))
    tgvars = Tuple(values(D))

    quote
        kwargs = LoopVectorization.nt($(QuoteNode(tgvars)), $(Expr(:tuple, tvars...)))
        $(Expr(:tuple, tvars...)) = LoopVectorization._avx($(QuoteNode(type_ex)), kwargs)
        # LoopVectorization._avx($(QuoteNode(type_ex)), kwargs) # comment out the above line, uncomment this one, and get rid of the `@generated` on _avx to see the function body.
    end |> esc
end

@generated function _avx(::Type{ex_t}, var_nt::NamedTuple{keys, var_types}) where {ex_t <: Ex, keys, var_types}
    ex = to_expr(ex_t)
    
    var_defs = Expr(:block, )
    for k in keys 
        push!(var_defs.args, :($k = var_nt[$(QuoteNode(k))]))
    end

    quote
        $(Expr(:meta,:inline))
        $var_defs
        $(lower(LoopSet(ex)))
        $(Expr(:tuple, keys...))
        #$(Expr(:tuple, (:($(keys[i]) :: $(var_types.parameters[i])) for i in eachindex(keys))...))
    end
end
