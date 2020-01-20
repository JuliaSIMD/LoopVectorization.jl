function Loop(::Type{UnitRange{Int}})
    Loop(gensym(:n), 0, 1024, gensym(:loopstart), gensym(:loopstop), false, false)::Loop
end
function Loop(::Type{StaticUpperUnitRange{U}}) where {U}
    Loop(gensym(:n), 0, U, gensym(:loopstart), Symbol(""), false, true)::Loop
end
function Loop(::Type{StaticLowerUnitRange{L}}) where {L}
    Loop(gensym(:n), L, L + 1024, Symbol(""), gensym(:loopstop), true, false)::Loop
end
function Loop(::Type{StaticUnitRange{L,U}}) where {L,U}
    Loop(gensym(:n), L, U, Symbol(""), Symbol(""), true, true)::Loop
end

function add_loops!(ls::LoopSet, LB)
    loopsyms = [gensym(:n) for _ ∈ eachindex(LB)]
    for l ∈ LB
        add_loop!(ls, Loop(LB)::Loop)
    end
    
end
function add_ops!(ls::LoopSet, ops::Vector{OperationStruct}, start::Int = 0, stopvptr = nothing)
    num_ops = length(ops)
    while start < num_ops
        start += 1
        opdescript = ops[start]
        
        stopvptr === vptr(op) && return start
    end
    0
end
numinds(u::UInt) = 8 - (leading_zeros(u) >>> 3)
function add_mref!(ls::LoopSet, ar::ArrayRef, arraysymbolinds::Vector{Symbol}, opsymbols::Vector{Symbol}, ::Type{PackedStridedPointer{T, N}}) where {T, N}
    index_types = ar.index_types
    indices = ar.indices
    ni = numinds(index_types)
    Ni = N + 1
    @assert ni == Ni
    index_vec = Vector{Symbol}(undef, Ni)
    while index_types != zero(UInt64)
        ind = indices % UInt8
        symind = if index_types == LoopIndex
            ls.loopsymbols[ind]
        elseif index_types == ComputedIndex
            opsymbols[ind]
        else
            @assert index_types == SymbolicIndex
            arraysymbolinds[ind] 
        end
        index_vec[ni] = symind
        index_types >>>= 8
        indices >>>= 8
        ni -= 1
    end
    
end

function add_mrefs!(ls::LoopSet, arf::Vector{ArrayRefStruct}, as::Vector{Symbol}, os::Vector{Symbol}, vargs)
    for i ∈ eachindex(arf)
        ref = arf[i]
        ptr_type = vargs[i]
        
    end
end
function process_metadata!(ls::LoopSet, AM, num_arrays::Int)
    num_asi = (AM[1])::Int
    arraysymbolinds = [gensym(:asi) for _ ∈ 1:num_asi]
    append!(ls.outer_reductions, AM[2].parameters)
    for (i,si) ∈ enumerate(AM[3].parameters)
        sii = si::Int
        s = gensym(:symlicm)
        push!(ls.preamble_symsym, (si,s))
        pushpreamble!(ls, Expr(:(=), s, Expr(:macrocall, Symbol("@inbounds"), LineNumberNode(@__LINE__,@__FILE__), Expr(:ref, :varg, num_arrays + i))))
    end
    append!(ls.preamble_symint, AM[4].parameters)
    append!(ls.preamble_symfloat, AM[5].parameters)
    append!(ls.preamble_zeros, AM[6].parameters)
    append!(ls.preamble_ones, AM[7].parameters)
    arraysymbolinds
end
function avx_body(ops, arf, AM, LB, vargs)
    ls = LoopSet()
    add_loops!(ls, LB)
    arraysymbolinds = process_metadata!(ls, AM, length(arf))
    opsymbols = [gensym(:op) for _ ∈ eachindex(ops)]
    
end

@generated function _avx!(::Type{OPS}, ::Type{ARF}, ::Type{AM}, lb::LB, vargs...)
    avx_body(
        OperationStruct[OPS.parameters...],
        ArrayRefStruct[ARF.parameters...],
        AM.parameters, LB.parameters, vargs
    )                    
end

