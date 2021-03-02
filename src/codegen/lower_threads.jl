
struct AVX{UNROLL,OPS,ARF,AM,LPSYM,LB,V} <: Function end

# This should call the same `_avx_!(Val{UNROLL}(), Val{OPS}(), Val{ARF}(), Val{AM}(), Val{LPSYM}(), _vargs)` as normal so that this
# hopefully shouldn't add much to compile time.

function (::AVX{UNROLL,OPS,ARF,AM,LPSYM,LB,V})(p::Ptr{UInt}) where {UNROLL,OPS,ARF,AM,LPSYM,LB,V}
    _vargs = ThreadingUtilities.load(p, Tuple{LB,V}, 1)
    ret = _avx_!(Val{UNROLL}(), Val{OPS}(), Val{ARF}(), Val{AM}(), Val{LPSYM}(), _vargs)
    ThreadingUtilities.store!(p, ret, 7)
    nothing
end

# function approx_cbrt(x)
#     s = significand(x)
#     e = exponent(x)
    
#     # 40 + 0.00020833333333333335*(x-64000)  -2.1701388888888896e-9*(x-64000)^2*0.5 + 5.6514033564814844e-14 * (x-64000)^3/6
# end

function choose_threads(::StaticInt{C}, x) where {C}
    nt = ifelse(gt(num_threads(), num_cores()), num_cores(), num_threads())
    fx = Base.uitofp(Float64, x)
    min(Base.fptosi(Int, Base.ceil_llvm(5.0852672001495816e-11*C*Base.sqrt_llvm(fx))), nt)
end

function thread_single_loop_expr(ls::LoopSet, UNROLL, id)

end
function thread_multiple_loop_expr(ls::LoopSet, UNROLL, valid_thread_loop)

end

function avx_threads_expr(ls::LoopSet, UNROLL)
    order, u₁loop, u₂loop, vectorized, u₁, u₂, c, shouldinline = choose_order_cost(ls)
    valid_thread_loop = fill(true, length(order))
    for op ∈ operations(ls)
        if isstore(op) && (length(reduceddependencies(op)) > 0)
            for reduceddep ∈ reduceddependencies(op)
                for (i,o) ∈ enumerate(order)
                    if o === reduceddep
                        valid_thread_loop[i] = false
                    end
                end
            end
        end
    end
    num_candiates = sum(valid_thread_loop)
    # num_to_thread = min(num_candiates, 2)
    # candidate_ids = 
    if num_candiates == 0
        avx_body(ls, UNROLL)
    elseif num_candiates == 1
        thread_single_loop_expr(ls, UNROLL, findfirst(isone, valid_thread_loop)::Int)
    else
        thread_multiple_loop_expr(ls, UNROLL, vald_thread_loop)
    end
    
end


