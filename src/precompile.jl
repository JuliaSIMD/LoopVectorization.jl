function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    precompile(Tuple{typeof(LoopVectorization.vectorize_body),Int64,Type{Float64},Int64,Symbol,Array{Any,1},Dict{Symbol,Tuple{Symbol,Symbol}},Any,Bool,Module})
    precompile(Tuple{typeof(LoopVectorization.vectorize_body),Int64,Type{Float64},Int64,Symbol,Array{Any,1},Dict{Symbol,Tuple{Symbol,Symbol}},Any,Bool})
    # precompile(Tuple{LoopVectorization.var"#_vectorloads!##kw",NamedTuple{(:itersym, :declared_iter_sym, :VectorizationDict, :mod),Tuple{Symbol,Symbol,Dict{Symbol,Tuple{Symbol,Symbol}},Module}},typeof(LoopVectorization._vectorloads!),Expr,Expr,Tuple{Dict{Symbol,Symbol},Dict{Tuple{Symbol,Symbol},Symbol},Dict{Expr,Symbol},Dict{Expr,Symbol}},Type,Int64,Type,Expr,Expr})
    precompile(Tuple{typeof(LoopVectorization.vectorize_body),Symbol,Type{Float64},Int64,Symbol,Array{Any,1},Dict{Symbol,Tuple{Symbol,Symbol}},Any,Bool})
    # precompile(Tuple{LoopVectorization.var"#_vectorloads!##kw",NamedTuple{(:itersym, :declared_iter_sym, :VectorizationDict, :mod),Tuple{Symbol,Symbol,Dict{Symbol,Tuple{Symbol,Symbol}},Symbol}},typeof(LoopVectorization._vectorloads!),Expr,Expr,Tuple{Dict{Symbol,Symbol},Dict{Tuple{Symbol,Symbol},Symbol},Dict{Expr,Symbol},Dict{Expr,Symbol}},Type,Int64,Type,Expr,Expr})
    precompile(Tuple{typeof(LoopVectorization.add_masks),Expr,Symbol,Dict{Tuple{Symbol,Symbol},Symbol},Module})
    precompile(Tuple{typeof(LoopVectorization.add_masks),Expr,Symbol,Dict{Tuple{Symbol,Symbol},Symbol},Symbol})
end
