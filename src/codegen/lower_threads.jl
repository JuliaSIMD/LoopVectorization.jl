struct AVX{UNROLL,OPS,ARF,AM,LPSYM,LB,V} <: Function end

# This should call the same `_avx_!(Val{UNROLL}(), Val{OPS}(), Val{ARF}(), Val{AM}(), Val{LPSYM}(), _vargs)` as normal so that this
# hopefully shouldn't add much to compile time.

function (::AVX{UNROLL,OPS,ARF,AM,LPSYM,LB,V})(p::Ptr{UInt}) where {UNROLL,OPS,ARF,AM,LPSYM,LB,V}
    (_, _vargs) = ThreadingUtilities.load(p, Tuple{LB,V}, 2*sizeof(UInt))
    # Main.VARGS[Threads.threadid()] = first(_vargs)
    ret = _avx_!(Val{UNROLL}(), Val{OPS}(), Val{ARF}(), Val{AM}(), Val{LPSYM}(), _vargs)
    ThreadingUtilities.store!(p, ret, Int(register_size()))
    nothing
end
@generated function Base.pointer(::AVX{UNROLL,OPS,ARF,AM,LPSYM,LB,V}) where {UNROLL,OPS,ARF,AM,LPSYM,LB,V}
    f = AVX{UNROLL,OPS,ARF,AM,LPSYM,LB,V}()
    precompile(f, (Ptr{UInt},))
    quote
        $(Expr(:meta,:inline))
        @cfunction($f, Cvoid, (Ptr{UInt},))
    end
end

@inline function setup_avx_threads!(p::Ptr{UInt}, fptr::Ptr{Cvoid}, args::Tuple{LB,V}) where {LB,V}
    offset = ThreadingUtilities.store!(p, fptr, sizeof(UInt))
    offset = ThreadingUtilities.store!(p, args, offset)
    nothing
end
@inline function avx_launch(
    ::Val{UNROLL}, ::Val{OPS}, ::Val{ARF}, ::Val{AM}, ::Val{LPSYM}, lb::LB, vargs::V, tid
) where {UNROLL,OPS,ARF,AM,LPSYM,LB,V}
    ThreadingUtilities.launch(setup_avx_threads!, tid, pointer(AVX{UNROLL,OPS,ARF,AM,LPSYM,LB,V}()), (lb,vargs))
end

# function approx_cbrt(x)
#     s = significand(x)
#     e = exponent(x)

#     # 40 + 0.00020833333333333335*(x-64000)  -2.1701388888888896e-9*(x-64000)^2*0.5 + 5.6514033564814844e-14 * (x-64000)^3/6
# end
lv_max_num_threads() = ifelse(gt(num_threads(), num_cores()), num_cores(), num_threads())

@generated function calc_factors(::StaticInt{nc}) where {nc}
    t = Expr(:tuple)
    for i ∈ nc:-1:1
        d, r = divrem(nc, i)
        iszero(r) && push!(t.args, (i % UInt, d % UInt))
    end
    t
end
@inline cld_fast(x,y) = Base.udiv_int(x + y - one(y), y)
@inline function choose_num_blocks(MoW::UInt, ::StaticInt{U}, ::StaticInt{T}) where {U,T}
    factors = calc_factors(StaticInt{T}())
    for i ∈ 1:length(factors)-1
        # miter decreases in each iteration of factors
        miter, niter = factors[i]

        r = (MoW % miter)
        # if ((miter * W * U * 2) ≤ M - (W+W)) & ((r == 0) | (miter == (r+1)))
        mlarge = (miter * (U * 2)) ≤ MoW - 2
        # we want `mlarge` enough, or there to be no remainder (`r == 0`)
        # or the remainder to be small relative to `M`, e.g. `M ÷ 128`.
        if mlarge && ((r == 0) || ((miter - r) ≤ (cld_fast(MoW, miter) >>> 3)))
            return miter, niter
        end
    end
    last(factors)
end
@inline function choose_num_blocks(::StaticInt{T}) where {T}
    factors = calc_factors(StaticInt{T}())
    @inbounds factors[(length(factors) + 1)>>>1]
end

# struct ChooseNumBlocks{U,C} <: Function end
# function (cnb::ChooseNumBlocks{U,C})(M::UInt) where {U,C}
#     choose_num_blocks(M, StaticInt{U}(), StaticInt{C}())
# end

# @generated function choose_num_block_table(::StaticInt{U}, ::StaticInt{NC}) where {U,NC}
#     t = Expr(:tuple)
#     for n ∈ 1:NC
#         cnb = :(ChooseNumBlocks{$U,$n}())
#         push!(t.args, :(@cfunction($cnb, Tuple{UInt,UInt}, (UInt,))))
#     end
#     t
# end
@generated function choose_num_block_table(::StaticInt{NC}) where {NC}
    t = Expr(:tuple)
    for n ∈ 1:NC
        push!(t.args, :(choose_num_blocks(StaticInt{$n}())))
    end
    t
end

@generated function _choose_num_blocks(M::UInt, ::StaticInt{U}, nt, ::StaticInt{NTMAX}) where {U,NTMAX}
    # valid range for nt: 2 ≤ nt ≤ NTMAX
    # if NTMAX > 8
    #     return quote
    #         $(Expr(:meta,:inline))
    #         choose_num_blocks_table(M, StaticInt{$U}(), nt, StaticInt{$NTMAX}())
    #     end
    # else
    if NTMAX == 2 # `nt` must be `2`
        return quote
            $(Expr(:meta,:inline))
            choose_num_blocks(M, StaticInt{$U}(), StaticInt{$NTMAX}())
        end
    end
    q = Expr(:block)#, Expr(:meta,:inline))
    ifq = Expr(
        :if,
        :(nt == $NTMAX),
        :(choose_num_blocks(M, StaticInt{$U}(), StaticInt{$NTMAX}()))
    )
    add_bisecting_if_branches!(ifq, 2, NTMAX-1, U, false)
    push!(q.args, ifq)
    q
end
function add_bisecting_if_branches!(q, lb, ub, U, isfirst::Bool)
    if lb == ub
        push!(q.args, :(choose_num_blocks(M, StaticInt{$U}(), StaticInt{$lb}())))
        return
    end
    midpoint = (lb + ub) >> 1
    alt = Expr(isfirst ? :if : :elseif, :(nt > $midpoint))
    add_bisecting_if_branches!(alt, midpoint+1, ub, U, true)
    add_bisecting_if_branches!(alt, lb, midpoint, U, false)
    push!(q.args, alt)
    return
end

# @inline function choose_num_blocks_table(M, ::StaticInt{U}, nt, ::StaticInt{NTMAX}) where {U,NTMAX}
#     if nt == NTMAX
#         choose_num_blocks(M % UInt, StaticInt{U}(), StaticInt{NTMAX}())
#     else
#         @inbounds fptr = choose_num_block_table(StaticInt{U}(), StaticInt{NTMAX}())[nt]
#         VectorizationBase.assume(fptr ≠ C_NULL)
#         ccall(fptr, Tuple{UInt,UInt}, (UInt,), M%UInt)
#     end
# end

# if a threaded loop is vectorized, call
@inline function choose_num_blocks(M, ::StaticInt{U}, nt) where {U}
    _choose_num_blocks(M % UInt, StaticInt{U}(), nt, lv_max_num_threads())
end
# otherwise, call
@inline choose_num_blocks(nt, ::StaticInt{NC} = lv_max_num_threads()) where {NC} = @inbounds choose_num_block_table(StaticInt{NC}())[nt]



# The goal is to minimimize the maximum costs...
# But maybe 'relatively even sizes' heuristics are more robust than fancy modeling?
# At least early on, before lots of test cases with different sorts of loops have informed the modeling.
#
# goal is to produce `nblocks` roughly even block sizes (bM, bN), such that `bM % fM == bN % fN == 0`.
# function roughly_even_blocks(M, N, fM, fN, nblocks)
#     M_N_ratio = M / N
#     block_per_m = sqrt(nblocks * M_N_ratio) # obv not even
#     blocks_per_n = block_per_m / M_N_ratio
#     mi = cld(M, fM)
#     ni = cld(N, fN)
#     block_per_m, blocks_per_n
# end

@inline choose_num_threads(C::Float64, NT::UInt, x::Base.BitInteger) = _choose_num_threads(Base.FastMath.mul_float_fast(C, 0.05460264079015985), NT, x)
@inline _choose_num_threads(C::Float64, NT::UInt, x::Base.BitInteger) = min(Base.fptoui(UInt, Base.ceil_llvm(Base.FastMath.mul_float_fast(C, Base.sqrt_llvm(Base.uitofp(Float64, x))))), NT)
function push_loop_length_expr!(q::Expr, ls::LoopSet)
    l = 1
    ndynamic = 0
    mulexpr = length(ls.loops) == 1 ? q : Expr(:call, lv(:vmul_fast))
    for loop ∈ ls.loops
        if isstaticloop(loop)
            l *= length(loop)
        else
            ndynamic += 1
            if ndynamic < 3
                push!(mulexpr.args, loop.lensym)
            else
                mulexpr = Expr(:call, lv(:vmul_fast), mulexpr, loop.lensym)
            end
        end
    end
    if length(ls.loops) == 1
        ndynamic == 0 && push!(q.args, l)
    elseif l == 1
        push!(q.args, mulexpr)
    elseif ndynamic == 0
        push!(q.args, l)
    elseif ndynamic == 1
        push!(mulexpr.args, l)
        push!(q.args, mulexpr)
    else
        push!(q.args, Expr(:call, :vmul_fast, mulexpr, l))
    end
    nothing
end
@inline function divrem_fast(numerator, denominator)
    d = Base.udiv_int(numerator, denominator)
    r = numerator - denominator*d
    d, r
end

function outer_reduct_combine_expressions(ls::LoopSet, retv)
    gf = GlobalRef(Core, :getfield)
    q = Expr(:block, :(var"#load#thread#ret#" = $gf(ThreadingUtilities.load(var"#thread#ptr#", typeof($retv), $(reg_size(ls))),2,false)))
    # push!(q.args, :(@show var"#load#thread#ret#"))
    for (i,or) ∈ enumerate(ls.outer_reductions)
        op = ls.operations[or]
        var = name(op)
        mvar = mangledvar(op)
        instr = instruction(op)
        out = Symbol(mvar, "##onevec##")
        instrcall = Expr(:call, lv(reduce_to_onevecunroll(instr)))
        push!(instrcall.args, Expr(:call, lv(:vecmemaybe), out))
        if length(ls.outer_reductions) > 1
            push!(instrcall.args, Expr(:call, lv(:vecmemaybe), Expr(:call, gf, Symbol("#load#thread#ret#"), i, false)))
        else
            push!(instrcall.args, Expr(:call, lv(:vecmemaybe), Symbol("#load#thread#ret#")))
        end
        push!(q.args, Expr(:(=), out, Expr(:call, :data, instrcall)))
    end
    q
end

function thread_loop_summary!(ls::LoopSet, ua::UnrollArgs, threadedloop::Loop, issecondthreadloop::Bool)
    W = ls.vector_width[]
    @unpack u₁loop, u₂loop, vloop, u₁, u₂max = ua
    u₂ = u₂max
    threadloopnumtag = Int(issecondthreadloop)
    lensym = Symbol("#len#thread#$threadloopnumtag#")
    define_len = if isstaticloop(threadedloop)
        :($lensym = $(length(threadedloop)) % UInt)
    else
        :($lensym = $((threadedloop.lensym)) % UInt)
    end
    unroll_factor = Core.ifelse(threadedloop === vloop, W, 1)
    # if threadedloop === u₁loop
    #     unroll_factor *= u₁
    # elseif threadedloop === u₂loop
    #     unroll_factor *= u₂
    # end
    num_unroll_sym = Symbol("#num#unrolls#thread#$threadloopnumtag#")
    define_num_unrolls = if unroll_factor == 1
        :($num_unroll_sym = $lensym)
    else
        # :($num_unroll_sym = Base.udiv_int($lensym, $(UInt(unroll_factor))))
        :($num_unroll_sym = Base.udiv_int($lensym + $(UInt(unroll_factor-1)), $(UInt(unroll_factor))))
    end
    iterstart_sym = Symbol("#iter#start#$threadloopnumtag#")
    iterstop_sym = Symbol("#iter#stop#$threadloopnumtag#")
    blksz_sym = Symbol("#nblock#size#thread#$threadloopnumtag#")
    loopstart = if isknown(first(threadedloop))
        :($iterstart_sym::Int = $(gethint(first(threadedloop))))
    else
        :($iterstart_sym::Int = $(getsym(first(threadedloop))))
    end
    if isknown(step(threadedloop))
        mf = gethint(step(threadedloop))
        if isone(mf)
            iterstop = :($iterstop_sym::Int = $iterstart_sym + $blksz_sym)
            looprange = :(CloseOpen($iterstart_sym))
            lastrange = :(CloseOpen($iterstart_sym))
            push_loopbound_ends!(looprange, lastrange, unroll_factor, threadedloop, iterstop_sym, true)
        else
            iterstop = :($iterstop_sym::Int = $iterstart_sym + $blksz_sym * $mf)
            looprange = :($iterstart_sym:StaticInt{$mf}())
            lastrange = :($iterstart_sym:StaticInt{$mf}())
            push_loopbound_ends!(looprange, lastrange, unroll_factor, threadedloop, :($iterstop_sym-one($iterstop_sym)), false)
        end
    else
        stepthread_sym = Symbol("#step#thread#$threadloopnumtag#")
        pushpreamble!(ls, :($stepthread_sym = $(getsym(step(threadedloop)))))
        iterstop = :($iterstop_sym = $iterstart_sym + $blksz_sym * $stepthread_sym)
        looprange = :($iterstart_sym:$stepthread_sym)
        lastrange = :($iterstart_sym:$stepthread_sym)
        push_loopbound_ends!(looprange, lastrange, unroll_factor, threadedloop, :($iterstop_sym-one($iterstop_sym)), false)
    end
    define_len, define_num_unrolls, loopstart, iterstop, looprange, lastrange
end
function push_last_bound!(looprange::Expr, lastrange::Expr, lastexpr, iterstop, unroll_factor::Int)
    push!(lastrange.args, lastexpr)
    unroll_factor ≠ 1 && push!(looprange.args, :(min($lastexpr, $iterstop)))
    nothing
end
function push_loopbound_ends!(
    looprange::Expr, lastrange::Expr, unroll_factor::Int,
    threadedloop::Loop, iterstop, offsetlast::Bool
)
    if unroll_factor == 1
        push!(looprange.args, iterstop)
    end
    if isknown(last(threadedloop))
        push_last_bound!(looprange, lastrange, gethint(last(threadedloop)) + offsetlast, iterstop, unroll_factor)
    else
        lastsym = getsym(last(threadedloop))
        if offsetlast
            push_last_bound!(looprange, lastrange, :($lastsym + one($lastsym)), iterstop, unroll_factor)
        else
            push_last_bound!(looprange, lastrange, lastsym, iterstop, unroll_factor)
        end
    end
    nothing
end
function define_block_size(threadedloop, vloop, tn, W)
    baseblocksizeuint = Symbol("#base#block#size#thread#uint#$tn#")
    baseblocksizeint = Symbol("#base#block#size#thread#$tn#")
    nrem = Symbol("#nrem#thread#$tn#")
    remstep = Symbol("#block#rem#step#$tn#")
    num_unroll = Symbol("#num#unrolls#thread#$tn#")
    thread_factor = Symbol("#thread#factor#$tn#")
    if threadedloop === vloop
        quote
            $baseblocksizeuint, $nrem = divrem_fast($num_unroll, $thread_factor)
            $baseblocksizeint = ($baseblocksizeuint << $(VectorizationBase.intlog2(W))) % Int
            $remstep = $(Int(W))
        end
    else
        quote
            $baseblocksizeuint, $nrem = divrem_fast($num_unroll, $thread_factor)
            $baseblocksizeint = $baseblocksizeuint % Int
            $remstep = 1
        end
    end
end
function thread_one_loops_expr(
    ls::LoopSet, ua::UnrollArgs, valid_thread_loop::Vector{Bool}, ntmax::UInt, c::Float64,
    UNROLL::Tuple{Bool,Int8,Int8,Bool,Int,Int,Int,Int,Int,Int,Int,UInt}, OPS::Expr, ARF::Expr, AM::Expr, LPSYM::Expr
)
    looplen = looplengthprod(ls)
    c = 0.05460264079015985 * c / looplen
    if all(isstaticloop, ls.loops)
        _num_threads = _choose_num_threads(c, ntmax, Int64(looplen))::UInt
        _num_threads > 1 || return avx_body(ls, UNROLL)
        choose_nthread = Expr(:(=), Symbol("#nthreads#"), _num_threads)
    else
        choose_nthread = :(_choose_num_threads($c, $ntmax))
        push_loop_length_expr!(choose_nthread, ls)
        choose_nthread = Expr(:(=), Symbol("#nthreads#"), choose_nthread)
    end
    threadedid = findfirst(valid_thread_loop)::Int
    threadedloop = getloop(ls, threadedid)
    define_len, define_num_unrolls, loopstart, iterstop, looprange, lastrange = thread_loop_summary!(ls, ua, threadedloop, false)
    loopboundexpr = Expr(:tuple) # for launched threads
    lastboundexpr = Expr(:tuple) # remainder, started on main thread
    for (i,loop) ∈ enumerate(ls.loops)
        if loop === threadedloop
            push!(loopboundexpr.args, looprange)
            push!(lastboundexpr.args, lastrange)
        else
            loop_boundary!(loopboundexpr, loop)
            loop_boundary!(lastboundexpr, loop)
        end
    end
    _avx_call_ = :(_avx_!(Val{$UNROLL}(), $OPS, $ARF, $AM, $LPSYM, ($lastboundexpr, var"#vargs#")))
    update_return_values = if length(ls.outer_reductions) > 0
        retv = loopset_return_value(ls, Val(false))
        _avx_call_ = Expr(:(=), retv, _avx_call_)
        outer_reduct_combine_expressions(ls, retv)
    else
        nothing
    end
    retexpr = length(ls.outer_reductions) > 0 ? :(return $retv) : :(return nothing)
    # @unpack u₁loop, u₂loop, vloop, u₁, u₂max = ua
    iterdef = define_block_size(threadedloop, ua.vloop, 0, ls.vector_width[])
    q = quote
        $choose_nthread # UInt
        $define_len
        $define_num_unrolls
        var"#nthreads#" = Base.min(var"#nthreads#", var"#num#unrolls#thread#0#")
        var"#nrequest#" = (var"#nthreads#" % UInt32) - 0x00000001
        $loopstart
        var"##do#thread##" = var"#nrequest#" ≠ 0x00000000
        if var"##do#thread##"
            var"#threads#", var"#torelease#" = CheapThreads.request_threads(Threads.threadid()%UInt32, var"#nrequest#")
            var"#thread#factor#0#" = var"#nthreads#"
            $iterdef
            var"#thread#launch#count#" = 0x00000000
            var"#thread#id#" = 0x00000000
            var"#thread#mask#" = CheapThreads.mask(var"#threads#")
            var"#threads#remain#" = true
            while var"#threads#remain#"
                VectorizationBase.assume(var"#thread#mask#" ≠ zero(var"#thread#mask#"))
                var"#trailzing#zeros#" = Base.trailing_zeros(var"#thread#mask#") % UInt32
                var"#nblock#size#thread#0#" = Core.ifelse(
                    var"#thread#launch#count#" < (var"#nrem#thread#0#" % UInt32),
                    var"#base#block#size#thread#0#" + var"#block#rem#step#0#",
                    var"#base#block#size#thread#0#"
                )
                var"#trailzing#zeros#" += 0x00000001
                $iterstop
                var"#thread#id#" += var"#trailzing#zeros#"

                avx_launch(
                    Val{$UNROLL}(), $OPS, $ARF, $AM, $LPSYM,
                    $loopboundexpr, var"#vargs#", var"#thread#id#"
                )

                var"#thread#mask#" >>>= var"#trailzing#zeros#"

                var"#iter#start#0#" = var"#iter#stop#0#"
                var"#threads#remain#" = (var"#thread#launch#count#" += 0x00000001) ≠ var"#nrequest#"
            end
        else# eliminate undef var errors that the compiler should be able to figure out are unreachable, but doesn't
            var"#torelease#" = zero(CheapThreads.worker_type())
            var"#threads#" = CheapThreads.UnsignedIteratorEarlyStop(var"#torelease#", 0x00000000)
        end
        $_avx_call_
        var"##do#thread##" || $retexpr
        var"#thread#id#" = 0x00000000
        var"#thread#mask#" = CheapThreads.mask(var"#threads#")
        var"#threads#remain#" = true
        while var"#threads#remain#"
            VectorizationBase.assume(var"#thread#mask#" ≠ zero(var"#thread#mask#"))
            var"#trailzing#zeros#" = Base.trailing_zeros(var"#thread#mask#") % UInt32
            var"#trailzing#zeros#" += 0x00000001
            var"#thread#mask#" >>>= var"#trailzing#zeros#"
            var"#thread#id#" += var"#trailzing#zeros#"
            var"#thread#ptr#" = ThreadingUtilities.taskpointer(var"#thread#id#")
            ThreadingUtilities.wait(var"#thread#ptr#")
            $update_return_values
            var"#threads#remain#" = var"#thread#mask#" ≠ 0x00000000
        end
        CheapThreads.free_threads!(var"#torelease#")
        $retexpr
    end
    Expr(:block, ls.preamble, q)
end
function define_vthread_blocks(vloop, u₁loop, u₂loop, u₁, u₂, ntmax, tn)
    loopunrollname = Symbol("#num#unrolls#thread#$tn#")
    lhs = tn == 0 ? :((var"#thread#factor#0#", var"#thread#factor#1#")) : :((var"#thread#factor#1#", var"#thread#factor#0#"))
    sntmax = staticexpr(ntmax % Int)
    if vloop === u₁loop
        :($lhs = _choose_num_blocks($loopunrollname, StaticInt{$u₁}(), var"#nthreads#", $sntmax))
    elseif vloop === u₂loop
        :($lhs = _choose_num_blocks($loopunrollname, StaticInt{$u₂}(), var"#nthreads#", $sntmax))
    else
        :($lhs  = _choose_num_blocks($loopunrollname, StaticInt{1}(), var"#nthreads#", $sntmax))
    end
end
function define_thread_blocks(threadedloop1, threadedloop2, vloop, u₁loop, u₂loop, u₁, u₂, ntmax)
    if vloop === threadedloop1
        define_vthread_blocks(threadedloop1, u₁loop, u₂loop, u₁, u₂, ntmax, 0)
    elseif vloop === threadedloop2
        define_vthread_blocks(threadedloop2, u₁loop, u₂loop, u₁, u₂, ntmax, 1)
    else
        :((var"#thread#factor#0#", var"#thread#factor#1#") = choose_num_blocks(var"#nthreads#", StaticInt{$(Int(ntmax))}()))
   end
end
function thread_two_loops_expr(
    ls::LoopSet, ua::UnrollArgs, valid_thread_loop::Vector{Bool}, ntmax::UInt, c::Float64,
    UNROLL::Tuple{Bool,Int8,Int8,Bool,Int,Int,Int,Int,Int,Int,Int,UInt}, OPS::Expr, ARF::Expr, AM::Expr, LPSYM::Expr
)
    looplen = looplengthprod(ls)
    c = 0.05460264079015985 * c / looplen
    if all(isstaticloop, ls.loops)
        _num_threads = _choose_num_threads(c, ntmax, Int64(looplen))::UInt
        _num_threads > 1 || return avx_body(ls, UNROLL)
        choose_nthread = Expr(:(=), Symbol("#nthreads#"), _num_threads)
    else
        choose_nthread = :(_choose_num_threads($c, $ntmax))
        push_loop_length_expr!(choose_nthread, ls)
        choose_nthread = Expr(:(=), Symbol("#nthreads#"), choose_nthread)
    end
    threadedid1 = threadedid2 = 0
    for (i,v) ∈ enumerate(valid_thread_loop)
        v || continue
        if threadedid1 == 0
            threadedid1 = i
        else#if threadedid2 == 0
            threadedid2 = i
            break
        end
    end
    @unpack u₁loop, u₂loop, vloop, u₁, u₂max = ua
    u₂ = u₂max
    W = ls.vector_width[]
    threadedloop1 = getloop(ls, threadedid1)
    threadedloop2 = getloop(ls, threadedid2)
    define_len1, define_num_unrolls1, loopstart1, iterstop1, looprange1, lastrange1 = thread_loop_summary!(ls, ua, threadedloop1, false)
    define_len2, define_num_unrolls2, loopstart2, iterstop2, looprange2, lastrange2 = thread_loop_summary!(ls, ua, threadedloop2, true)
    loopboundexpr = Expr(:tuple)
    lastboundexpr = Expr(:tuple)
    for (i,loop) ∈ enumerate(ls.loops)
        if loop === threadedloop1
            push!(loopboundexpr.args, looprange1)
            push!(lastboundexpr.args, lastrange1)
        elseif loop === threadedloop2
            push!(loopboundexpr.args, looprange2)
            push!(lastboundexpr.args, lastrange2)
        else
            loop_boundary!(loopboundexpr, loop)
            loop_boundary!(lastboundexpr, loop)
        end
    end
    _avx_call_ = :(_avx_!(Val{$UNROLL}(), $OPS, $ARF, $AM, $LPSYM, ($lastboundexpr, var"#vargs#")))
    update_return_values = if length(ls.outer_reductions) > 0
        retv = loopset_return_value(ls, Val(false))
        _avx_call_ = Expr(:(=), retv, _avx_call_)
        outer_reduct_combine_expressions(ls, retv)
    else
        nothing
    end
    blockdef = define_thread_blocks(threadedloop1, threadedloop2, vloop, u₁loop, u₂loop, u₁, u₂, ntmax)
    iterdef1 = define_block_size(threadedloop1, vloop, 0, ls.vector_width[])
    iterdef2 = define_block_size(threadedloop2, vloop, 1, ls.vector_width[])
    retexpr = length(ls.outer_reductions) > 0 ? :(return $retv) : :(return nothing)
    q = quote
        $choose_nthread # UInt
        $define_len1
        $define_len2
        $define_num_unrolls1
        $define_num_unrolls2
        var"#unroll#prod#" = var"#num#unrolls#thread#0#" * var"#num#unrolls#thread#1#"
        if var"#nthreads#" ≥ var"#unroll#prod#"
            var"#nthreads#" = var"#unroll#prod#"
            var"#thread#factor#0#" = var"#num#unrolls#thread#0#"
            var"#thread#factor#1#" = var"#num#unrolls#thread#1#"
        else
            $blockdef
        end
        # @show (var"#thread#factor#0#",var"#thread#factor#1#")
        var"#nrequest#" = (var"#nthreads#" % UInt32) - 0x00000001
        $loopstart1
        var"#loop#1#start#init#" = var"#iter#start#0#"
        $loopstart2
        var"##do#thread##" = var"#nrequest#" ≠ 0x00000000
        if var"##do#thread##"
            var"#threads#", var"#torelease#" = CheapThreads.request_threads(Threads.threadid(), var"#nrequest#")
            $iterdef1
            $iterdef2
            # @show var"#base#block#size#thread#0#", var"#block#rem#step#0#" var"#base#block#size#thread#1#", var"#block#rem#step#1#"
            var"#thread#launch#count#" = 0x00000000
            var"#thread#launch#count#0#" = 0x00000000
            var"#thread#launch#count#1#" = 0x00000000
            var"#thread#id#" = 0x00000000
            var"#thread#mask#" = CheapThreads.mask(var"#threads#")
            var"#threads#remain#" = true
            while var"#threads#remain#"
                VectorizationBase.assume(var"#thread#mask#" ≠ zero(var"#thread#mask#"))
                var"#trailzing#zeros#" = Base.trailing_zeros(var"#thread#mask#") % UInt32
                var"#nblock#size#thread#0#" = Core.ifelse(
                    var"#thread#launch#count#0#" < (var"#nrem#thread#0#" % UInt32),
                    var"#base#block#size#thread#0#" + var"#block#rem#step#0#",
                    var"#base#block#size#thread#0#"
                )
                var"#nblock#size#thread#1#" = Core.ifelse(
                    var"#thread#launch#count#1#" < (var"#nrem#thread#1#" % UInt32),
                    var"#base#block#size#thread#1#" + var"#block#rem#step#1#",
                    var"#base#block#size#thread#1#"
                )
                var"#trailzing#zeros#" += 0x00000001
                $iterstop1
                $iterstop2
                var"#thread#id#" += var"#trailzing#zeros#"
                # @show var"#thread#id#" $loopboundexpr
                avx_launch(
                    Val{$UNROLL}(), $OPS, $ARF, $AM, $LPSYM,
                    $loopboundexpr, var"#vargs#", var"#thread#id#"
                )

                var"#thread#mask#" >>>= var"#trailzing#zeros#"

                var"##end#inner##" = var"#thread#launch#count#0#" == (var"#thread#factor#0#"-0x00000001)
                var"#thread#launch#count#0#" = Core.ifelse(var"##end#inner##", 0x00000000, var"#thread#launch#count#0#" + 0x00000001)
                var"#thread#launch#count#1#" = Core.ifelse(var"##end#inner##", var"#thread#launch#count#1#" + 0x00000001, var"#thread#launch#count#1#")

                var"#iter#start#0#" = Core.ifelse(var"##end#inner##", var"#loop#1#start#init#", var"#iter#stop#0#")
                var"#iter#start#1#" = Core.ifelse(var"##end#inner##", var"#iter#stop#1#", var"#iter#start#1#")

                var"#threads#remain#" = (var"#thread#launch#count#" += 0x00000001) ≠ var"#nrequest#"
            end
        else# eliminate undef var errors that the compiler should be able to figure out are unreachable, but doesn't
            var"#torelease#" = zero(CheapThreads.worker_type())
            var"#threads#" = CheapThreads.UnsignedIteratorEarlyStop(var"#torelease#", 0x00000000)
        end
        # @show $lastboundexpr
        $_avx_call_
        var"##do#thread##" || $retexpr
        # @show $retv
        var"#thread#id#" = 0x00000000
        var"#thread#mask#" = CheapThreads.mask(var"#threads#")
        var"#threads#remain#" = true
        while var"#threads#remain#"
            VectorizationBase.assume(var"#thread#mask#" ≠ zero(var"#thread#mask#"))
            var"#trailzing#zeros#" = Base.trailing_zeros(var"#thread#mask#") % UInt32
            var"#trailzing#zeros#" += 0x00000001
            var"#thread#mask#" >>>= var"#trailzing#zeros#"
            var"#thread#id#" += var"#trailzing#zeros#"
            var"#thread#ptr#" = ThreadingUtilities.taskpointer(var"#thread#id#")
            ThreadingUtilities.wait(var"#thread#ptr#")
            $update_return_values
            var"#threads#remain#" = var"#thread#mask#" ≠ 0x00000000
        end
        CheapThreads.free_threads!(var"#torelease#")
        $retexpr
    end
    # @show
    Expr(:block, ls.preamble, q)
end

function valid_thread_loops(ls::LoopSet)
    order, u₁loop, u₂loop, vectorized, u₁, u₂, c, shouldinline = choose_order_cost(ls)
    # NOTE: `names` are being placed in the opposite order here versus normal lowering!
    copyto!(names(ls), order); init_loop_map!(ls)
    u₁loop = getloop(ls, u₁loop)
    _u₂loop = getloopid_or_nothing(ls, u₂loop)
    u₂loop = _u₂loop === nothing ? u₁loop : getloop_from_id(ls, _u₂loop)
    ua = UnrollArgs(u₁loop, u₂loop, getloop(ls, vectorized), u₁, u₂, u₂)
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
    valid_thread_loop, ua, c
end
function avx_threads_expr(
    ls::LoopSet, UNROLL::Tuple{Bool,Int8,Int8,Bool,Int,Int,Int,Int,Int,Int,Int,UInt},
    nt::UInt, OPS::Expr, ARF::Expr, AM::Expr, LPSYM::Expr
)
    valid_thread_loop, ua, c = valid_thread_loops(ls)
    num_candiates = sum(valid_thread_loop)
    # num_to_thread = min(num_candiates, 2)
    # candidate_ids =
    if (num_candiates == 0) || (nt ≤ 1) # it was called from `avx_body` but now `nt` was set to `1`
        avx_body(ls, UNROLL)
    elseif (num_candiates == 1) || (nt ≤ 3)
        thread_one_loops_expr(ls, ua, valid_thread_loop, nt, c, UNROLL, OPS, ARF, AM, LPSYM)
    else # requires at least 4 threads
        thread_two_loops_expr(ls, ua, valid_thread_loop, nt, c, UNROLL, OPS, ARF, AM, LPSYM)
    end
end
