
struct AVX{UNROLL,OPS,ARF,AM,LPSYM,LB,V} <: Function end

# This should call the same `_avx_!(Val{UNROLL}(), Val{OPS}(), Val{ARF}(), Val{AM}(), Val{LPSYM}(), _vargs)` as normal so that this
# hopefully shouldn't add much to compile time.
function (::AVX{UNROLL,OPS,ARF,AM,LPSYM,LB,V})(p::Ptr{UInt}) where {UNROLL,OPS,ARF,AM,LPSYM,LB,V}
    _vargs = ThreadingUtilities.load(p, Tuple{LB,V}, 1)
    ret = _avx_!(Val{UNROLL}(), Val{OPS}(), Val{ARF}(), Val{AM}(), Val{LPSYM}(), _vargs)
    ThreadingUtilities.store!(p, ret, 7)
    nothing
end



function _avx_threads!()
    
end


