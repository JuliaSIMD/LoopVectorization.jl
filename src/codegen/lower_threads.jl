
struct AVX{UNROLL,OPS,ARF,AM,LPSYM,LB,V} <: Function end

# This should call the same `_avx_!(Val{UNROLL}(), Val{OPS}(), Val{ARF}(), Val{AM}(), Val{LPSYM}(), _vargs)` as normal so that this
# hopefully shouldn't add much to compile time.

function (::AVX{UNROLL,OPS,ARF,AM,LPSYM,LB,V})(p::Ptr{UInt}) where {UNROLL,OPS,ARF,AM,LPSYM,LB,V}
    _vargs = ThreadingUtilities.load(p, Tuple{LB,V}, 1)
    ret = _avx_!(Val{UNROLL}(), Val{OPS}(), Val{ARF}(), Val{AM}(), Val{LPSYM}(), _vargs)
    ThreadingUtilities.store!(p, ret, 7)
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

function launch!(p::Ptr{UInt}, fptr::Ptr{Cvoid}, args::Tuple{LB,V}) where {LB,V}
    offset = ThreadingUtilities.store!(p, fptr, 0)
    offset = ThreadingUtilities.store!(p, args, offset)
    nothing
end
function launch(
    ::Val{UNROLL}, ::Val{OPS}, ::Val{ARF}, ::Val{AM}, ::Val{LPSYM}, lb::LB, vargs::V, tid
) where {UNROLL,OPS,ARF,AM,LPSYM,LB,V}
    p = ThreadingUtilities.taskpointer(tid)
    f = AVX{UNROLL,OPS,ARF,AM,LPSYM,LB,V}()
    fptr = pointer(f)
    while true
        if ThreadingUtilities._atomic_cas_cmp!(p, ThreadingUtilities.SPIN, ThreadingUtilities.STUP)
            launch!(p, fptr, (lb,vargs))
            @assert ThreadingUtilities._atomic_cas_cmp!(p, ThreadingUtilities.STUP, ThreadingUtilities.TASK)
            return
        elseif ThreadingUtilities._atomic_cas_cmp!(p, ThreadingUtilities.WAIT, ThreadingUtilities.STUP)
            launch!(p, fptr, (lb,vargs))
            @assert ThreadingUtilities._atomic_cas_cmp!(p, ThreadingUtilities.STUP, ThreadingUtilities.LOCK)
            ThreadingUtilities.wake_thread!(tid % UInt)
            return
        end
        ThreadingUtilities.pause()
    end
end

# function approx_cbrt(x)
#     s = significand(x)
#     e = exponent(x)
    
#     # 40 + 0.00020833333333333335*(x-64000)  -2.1701388888888896e-9*(x-64000)^2*0.5 + 5.6514033564814844e-14 * (x-64000)^3/6
# end

function choose_num_threads(::StaticInt{C}, x) where {C}
    nt = ifelse(gt(num_threads(), num_cores()), num_cores(), num_threads())
    fx = Base.uitofp(Float64, x)
    min(Base.fptoui(UInt, Base.ceil_llvm(5.0852672001495816e-11*C*Base.sqrt_llvm(fx))), UInt(nt))
end
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
function divrem_fast(numerator, denominator)
    d = Base.udiv_int(numerator, denominator)
    r = numerator - denominator*d
    d, r
end

function outer_reduct_combine_expressions(ls::LoopSet, retv)
    q = Expr(:block, :(var"#load#thread#ret#" = ThreadingUtilities.store!(var"#thread#ptr#", typeof($retv), 7)))
    for (i,or) ∈ enumerate(ls.outer_reductions)
        op = ls.operations[or]
        var = name(op)
        mvar = mangledvar(op)
        instr = instruction(op)
        out = Symbol(mvar, "##onevec##")
        instrcall = callexp(instr)
        push!(instrcall.args, Expr(:call, lv(:vecmemaybe), out))
        if length(ls.outer_reductions) > 1
            push!(instrcall.args, Expr(:call, lv(:vecmemaybe), Expr(:call, GlobalRef(Core, :getfield), Symbol("#load#thread#ret#"), i, false)))
        else
            push!(instrcall.args, Expr(:call, lv(:vecmemaybe), Symbol("#load#thread#ret#")))
        end
        push!(q.args, Expr(:(=), out, Expr(:call, :data, instrcall)))
    end
    q
end

function thread_loop_summary!(ls, threadedloop::Loop, u₁loop::Loop, u₂loop::Loop, vloop::Loop, issecondthreadloop::Bool)
    threadloopnumtag = Int(issecondthreadloop)
    lensym = Symbol("#len#thread#$threadloopnumtag#")
    define_len = if isstaticloop(threadedloop)
        :($lensym = $(length(threadedloop)))
    else
        :($lensym = $((threadedloop.lensym)))
    end
    unroll_factor = 1
    if threadedloop === vloop
        unroll_factor *= W
    end
    if threadedloop === u₁loop
        unroll_factor *= u₁
    elseif threadedloop === u₂loop
        unroll_factor *= u₂
    end
    num_unroll_sym = Symbol("#num#unrolls#thread#$threadloopnumtag#")
    define_num_unrolls = if unroll_factor == 1
        :($num_unroll_sym = $lensym)
    else
        :($num_unroll_sym = Base.udiv_int($lensym, $(UInt(unroll_factor))))
    end
    iterstart_sym = Symbol("#iter#start#$threadloopnumtag#")
    iterstop_sym = Symbol("#iter#stop#$threadloopnumtag#")
    blksz_sym = Symbol("#nblock#size#thread#$threadloopnumtag#")
    loopstart = if isknown(first(threadedloop))
        :($iterstart_sym = $(gethint(first(threadedloop))))
    else
        :($iterstart_sym = $(getsym(first(threadedloop))))
    end
    if isknown(step(threadedloop))
        mf = gethint(threadedloop) * unroll_factor
        if isone(mf)
            iterstop = :($iterstop_sym = $iterstart_sym + $blksz_sym)
            looprange = :(CloseOpen($iterstart_sym, $iterstop_sym))
            lastrange = if isknown(last(threadedloop))
                :(CloseOpen($iterstart_sym,$(gethint(threadedloop)+1)))
            else # we want all the intervals to have the same type.
                :(CloseOpen($iterstart_sym,$(getsym(threadedloop))+1))
            end
        else
            iterstop = :($iterstop_sym = $iterstart_sym + $blksz_sym * $mf)
            looprange = :($iterstart_sym:StaticInt{$mf}():$iterstop_sym-1)
            lastrange = if isknown(last(threadedloop))
                :($iterstart_sym:StaticInt{$mf}():$(gethint(threadedloop)))
            else
                :($iterstart_sym:StaticInt{$mf}():$(getsym(threadedloop)))
            end
        end
    else
        stepthread_sym = Symbol("#step#thread#$threadloopnumtag#")
        pushpreamble!(ls, :($stepthread_sym = $unroll_factor * $(getsym(step(threadedloop)))))
        iterstop = :($iterstop_sym = $iterstart_sym + $blksz_sym * $stepthread_sym)
        looprange = :($iterstart_sym:$stepthread_sym:$iterstop_sym-1)
        lastrange = if isknown(last(threadedloop))
            :($iterstart_sym:$stepthread_sym:$(gethint(threadedloop)))
        else
            :($iterstart_sym:$stepthread_sym:$(getsym(threadedloop)))
        end
    end
    define_len, define_num_unrolls, loopstart, iterstop, looprange, lastrange
end

function thread_single_loop_expr(ls::LoopSet, ua::UnrollArgs, valid_thread_loop, c, UNROLL, OPS, ARF, AM, LPSYM)
    choose_nthread = :(choose_num_threads(StaticInt{$c}()))
    push_loop_length_expr!(choose_nthread, ls)
    threadedid = findfirst(valid_thread_loop)::Int
    @unpack u₁loop, u₂loop, vloop, u₁, u₂ = ua
    W = ls.vector_width[]
    threadedloop = getloop(ls, threadedid)
    define_len, define_num_unrolls, loopstart, iterstop, looprange, lastrange = thread_loop_summary!(ls, threadedloop, u₁loop, u₂loop, vloop, 0)
    loopboundexpr = Expr(:tuple)
    lastboundexpr = Expr(:tuple)
    for (i,loop) ∈ enumerate(threadedloop)
        if loop === threadedloop
            push!(loopboundexpr.args, looprange)
            push!(lastboundexpr.args, lastrange)
        else
            loop_boundary!(loopboundexpr, loop)
            loop_boundary!(lastboundexpr, loop)
        end
    end
    _avx_call_ = :(_avx_!(Val{$UNROLL}(), Val{$OPS}(), Val{$ARF}(), Val{$AM}(), Val{$LPSYM}(), $lastboundexpr, var"#vargs#"))
    update_return_values = if length(ls.outer_reductions) > 0
        retv = loopset_return_value(ls, Val(false))
        _avx_call_ = Expr(:(=), retv, _avx_call_)
        outer_reduct_combine_expressions(ls, retv)
    else
        nothing
    end
    q = quote
        var"#nthreads#" = $choose_nthread # UInt
        $define_len % UInt
        $define_num_unrolls
        var"#nthreads#" = Base.min(var"#nthreads#", $num_unrolls)
        var"#nrequest#" = (var"#nthreads#" % UInt32) - 0x00000001
        var"#nrequest#" == 0x00000000 && return LoopVectorization._avx_!(Val{$UNROLL}(), Val{$OPS}, Val{$ARF}(), Val{$AM}(), Val{$LPSYM}(), var"#lv#tuple#args#")
        var"#threads#", var"#torelease#" = LoopVectorization._request_threads(Threads.threadid(), var"#nrequest#")

        var"#base#block#size#thread#0#", var"#nrem#thread#" = LoopVectorization.divrem_fast(num_unrolls, var"#nthreads#")
        $loopstart
        
        var"#thread#launch#count#" = 0x00000000
        var"#thread#id#" = 0x00000000
        var"#thread#mask#" = CheapThreads.mask(var"#threads#")
        var"#threads#remain#" = true
        while var"#threads#remain#"
            VectorizationBase.assume(var"#thread#mask#" ≠ zero(var"#thread#mask#"))
            var"#trailzing#zeros#" = Base.trailing_zeros(var"#thread#mask#") % UInt32
            var"#thread#launch#count#" += 0x00000001
            var"#nblock#size#thread#0#" = Core.ifelse(
                var"#thread#launch#count#" < (var"#nrem#thread#" % Base.typeof(var"#threadid#")),
                var"#base#block#size#thread#0#" + Base.one(var"#base#block#size#thread#0#"),
                var"#base#block#size#thread#0#"
            )
            var"#trailzing#zeros#" += 0x00000001
            $iterstop
            var"#thread#id#" += var"#trailzing#zeros#"
            
            LoopVectorization.launch(
                Val{$UNROLL}(), Val{$OPS}(), Val{$ARF}(), Val{$AM}(), Val{$LPSYM}(),
                $loopboundexpr, var"#vargs#", var"#thread#id#"
            )
            
            var"#thread#mask#" >>>= var"#trailzing#zeros#"
            
            var"#iter#start#0#" = var"#iter#stop#0#"
            var"#threads#remain#" = var"#thread#launch#count#" ≠ var"$nrequest#"
        end
        $_avx_call_
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
            ThreadingUtilities.__wait(var"#thread#ptr#")
            $update_return_values
            var"#threads#remain#" = var"#thread#mask#" ≠ 0x00000000
        end
        CheapThreads.free_threads!(var"#torelease#")
    end
    length(ls.outer_reductions) > 0 ? push!(q.args, retv) : push!(q.args, nothing)
    q
end
function thread_multiple_loop_expr(ls::LoopSet, UNROLL, valid_thread_loop)

end

function valid_thread_loops(ls::LoopSet)
    order, u₁loop, u₂loop, vectorized, u₁, u₂, c, shouldinline = choose_order_cost(ls)
    # NOTE: `names` are being placed in the opposite order here versus normal lowering!
    copyto!(names(ls), order); init_loop_map!(ls)
    ua = UnrollArgs(getloop(ls, u₁loop), getloop(ls, u₂loop), getloop(ls, vloop), u₁, u₂, u₂)
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
function avx_threads_expr(ls::LoopSet, UNROLL)
    valid_thread_loop, us, c = valid_thread_loops(ls)
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


