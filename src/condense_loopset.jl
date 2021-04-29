@enum IndexType::UInt8 NotAnIndex=0 LoopIndex=1 ComputedIndex=2 SymbolicIndex=3

Base.:|(u::Unsigned, it::IndexType) = u | UInt8(it)
Base.:(==)(u::Unsigned, it::IndexType) = (u % UInt8) == UInt8(it)

"""
    ArrayRefStruct

A condensed representation of an [`ArrayReference`](@ref).
It supports array-references with up to 8 indexes, where the data for each consecutive index is packed into corresponding 8-bit fields
of `index_types` (storing the enum `IndexType`), `indices` (the `id` for each index symbol), and `offsets` (currently unused).
"""
struct ArrayRefStruct{array,ptr}
    index_types::UInt64
    indices::UInt64
    offsets::UInt64
    strides::UInt64
end
array_and_ptr(@nospecialize(ar::ArrayRefStruct{a,p})) where {a,p} = (a::Symbol,p::Symbol)
# array(@nospecialize(ar::ArrayRefStruct{a,p})) where {a,p} = a::Symbol
# ptr(@nospecialize(ar::ArrayRefStruct{a,p})) where {a,p}   = p::Symbol

function findindoradd!(v::Vector{T}, s::T) where {T}
    ind = findfirst(==(s), v)
    ind === nothing || return ind
    push!(v, s)
    length(v)
end
function ArrayRefStruct(ls::LoopSet, mref::ArrayReferenceMeta, arraysymbolinds::Vector{Symbol}, ids::Vector{Int})
    index_types = zero(UInt64)
    indices = zero(UInt64)
    offsets = zero(UInt64)
    strides = zero(UInt64)
    @unpack loopedindex, ref = mref
    indv = ref.indices
    offv = ref.offsets
    strv = ref.strides
    # we can discard that the array was considered discontiguous, as it should be recovered from type information
    start = 1 + (first(indv) === DISCONTIGUOUS)
    for (n,ind) ∈ enumerate(@view(indv[start:end]))
        index_types <<= 8
        indices <<= 8
        offsets <<= 8
        offsets |= (offv[n] % UInt8)
        strides <<= 8
        strides |= (strv[n] % UInt8)
        if loopedindex[n]
            index_types |= LoopIndex
            if strv[n] ≠ 0
                indices |= getloopid(ls, ind)
            end
        else
            parent = get(ls.opdict, ind, nothing)
            @assert !(parent === nothing) "Index $ind not found in array."
            # if parent === nothing
            #     index_types |= SymbolicIndex
            #     indices |= findindoradd!(arraysymbolinds, ind)
            # else
            index_types |= ComputedIndex
            indices |= ids[identifier(parent)]
            # end
        end
    end
    ArrayRefStruct{mref.ref.array,mref.ptr}( index_types, indices, offsets, strides )
end

"""
    OperationStruct

A condensed representation of an [`Operation`](@ref).
"""
struct OperationStruct <: AbstractLoopOperation
    # instruction::Instruction
    loopdeps::UInt64
    reduceddeps::UInt64
    childdeps::UInt64
    parents::UInt64
    node_type::OperationType
    array::UInt8
    symid::UInt8
end
optype(os) = os.node_type

function findmatchingarray(ls::LoopSet, mref::ArrayReferenceMeta)
    id = 0x01
    for r ∈ ls.refs_aliasing_syms
        r == mref && return id
        id += 0x01
    end
    0x00
end
# filled_4byte_chunks(u::UInt64) = 16 - (leading_zeros(u) >>> 2)
filled_8byte_chunks(u::UInt64) = 8 - (leading_zeros(u) >>> 3)

# num_loop_deps(os::OperationStruct) = filled_4byte_chunks(os.loopdeps)
# num_reduced_deps(os::OperationStruct) = filled_4byte_chunks(os.reduceddeps)
# num_child_deps(os::OperationStruct) = filled_4byte_chunks(os.childdeps)
# num_parents(os::OperationStruct) = filled_4byte_chunks(os.parents)

function shifted_loopset(ls::LoopSet, loopsyms::Vector{Symbol})
    ld = zero(UInt64) # leading_zeros(ld) >> 2 yields the number of loopdeps
    for d ∈ loopsyms
        ld <<= 4
        ld |= getloopid(ls, d)::Int
    end
    ld
end
# loopdeps_uint(ls::LoopSet, op::Operation) = (@show op; shifted_loopset(ls, loopdependencies(op)))
loopdeps_uint(ls::LoopSet, op::Operation) = shifted_loopset(ls, loopdependencies(op))
reduceddeps_uint(ls::LoopSet, op::Operation) = shifted_loopset(ls, reduceddependencies(op))
childdeps_uint(ls::LoopSet, op::Operation) = shifted_loopset(ls, reducedchildren(op))
function parents_uint(ls::LoopSet, op::Operation)
    p = zero(UInt64)
    for parent ∈ parents(op)
        p <<= 8
        p |= identifier(parent)
    end
    p
end
function recursively_set_parents_true!(x::Vector{Bool}, op::Operation)
    x[identifier(op)] && return nothing # don't redescend
    x[identifier(op)] = true
    for opp ∈ parents(op)
        recursively_set_parents_true!(x, opp)
    end
    return nothing
end
function getroots(ls::LoopSet)::Vector{Bool}
    rooted = Vector{Bool}(undef, length(operations(ls)))
    getroots!(rooted, ls)
end
function getroots!(rooted::Vector{Bool}, ls::LoopSet)
    fill!(rooted, false)
    ops = operations(ls)
    for or ∈ ls.outer_reductions
        recursively_set_parents_true!(rooted, ops[or])
    end
    for op ∈ ops
        isstore(op) && recursively_set_parents_true!(rooted, op)
    end
    return rooted
end
function OperationStruct!(varnames::Vector{Symbol}, ids::Vector{Int}, ls::LoopSet, op::Operation)
    instr = instruction(op)
    ld = loopdeps_uint(ls, op)
    rd = reduceddeps_uint(ls, op)
    cd = childdeps_uint(ls, op)
    p = parents_uint(ls, op)
    array = accesses_memory(op) ? findmatchingarray(ls, op.ref) : 0x00
    ids[identifier(op)] = id = findindoradd!(varnames, name(op))
    OperationStruct(
        ld, rd, cd, p, op.node_type, array, id
    )
end
## turn a LoopSet into a type object which can be used to reconstruct the LoopSet.

@inline zerorangestart(r::Base.OneTo) = CloseOpen(length(r))
@inline zerorangestart(r::CloseOpen) = CloseOpen(length(r))
@inline zerorangestart(r::AbstractUnitRange) = Zero():One():(maybestaticlast(r)-maybestaticfirst(r))
@inline zerorangestart(r::AbstractRange) = Zero():static_step(r):(maybestaticlast(r)-maybestaticfirst(r))
@inline zerorangestart(r::CartesianIndices) = CartesianIndices(map(zerorangestart, r.indices))

function loop_boundary!(q::Expr, ls::LoopSet, loop::Loop, shouldindbyind::Bool)
  if isstaticloop(loop) || loop.rangesym === Symbol("")
    call = Expr(:call, :(:))
    f = gethint(first(loop))
    s = gethint(step(loop))
    l = gethint(last(loop))
    if !shouldindbyind
      l -= f
      f = 0
    end
    pushexpr!(call, staticexpr(f))
    isone(s) || pushexpr!(call, staticexpr(s))
    pushexpr!(call, staticexpr(l))
    push!(q.args, call)
  elseif shouldindbyind
    push!(q.args, loop.rangesym)
  else
    push!(q.args, Expr(:call, lv(:zerorangestart), loop.rangesym))
  end
end

function loop_boundaries(ls::LoopSet, shouldindbyind::Vector{Bool})
    lbd = Expr(:tuple)
    for (ibi, loop) ∈ zip(shouldindbyind, ls.loops)
        loop_boundary!(lbd, ls, loop, ibi)
    end
    lbd
end

tuple_expr(v) = tuple_expr(identity, v)
function tuple_expr(f, v)
    t = Expr(:tuple)
    for vᵢ ∈ v
        push!(t.args, f(vᵢ))
    end
    t
end

function argmeta_and_consts_description(ls::LoopSet, arraysymbolinds)
    Expr(
        :tuple,
        length(arraysymbolinds),
        tuple_expr(ls.outer_reductions),
        tuple_expr(first, ls.preamble_symsym),
        tuple_expr(ls.preamble_symint),
        tuple_expr(ls.preamble_symfloat),
        tuple_expr(ls.preamble_zeros),
        tuple_expr(ls.preamble_funcofeltypes)
    )
end
@inline vdata(v::Vec) = getfield(v, :data)
@inline vdata(v::VecUnroll) = getfield(v, :data)
@inline vdata(x) = x

function loopset_return_value(ls::LoopSet, ::Val{extract}) where {extract}
    @assert !iszero(length(ls.outer_reductions))
    if isone(length(ls.outer_reductions))
        op = getop(ls, ls.outer_reductions[1])
        if extract
            # if (isu₁unrolled(op) | isu₂unrolled(op))
                Expr(:call, :vdata, Symbol(mangledvar(op), "##onevec##"))
            # else
                # Expr(:call, :data, mangledvar(op))
            # end
        else
            Symbol(mangledvar(op), "##onevec##")
        end
    else#if length(ls.outer_reductions) > 1
        ret = Expr(:tuple)
        ops = operations(ls)
        for or ∈ ls.outer_reductions
            op = ops[or]
            if extract
                push!(ret.args, Expr(:call, :vdata, Symbol(mangledvar(op), "##onevec##")))
            else
                push!(ret.args, Symbol(mangledvar(ops[or]), "##onevec##"))
            end
        end
        ret
    end
end
const DROPPEDCONSTANT = Instruction(Symbol("##DROPPED#CONSTANT##"),Symbol("##DROPPED#CONSTANT##"))
function skip_constant(instr::Instruction)
    ((instr == LOOPCONSTANT) || (instr.mod === :numericconstant)) || (instr == DROPPEDCONSTANT)
end

function add_reassigned_syms!(q::Expr, ls::LoopSet)
    for op ∈ operations(ls)
        if isconstant(op)
            instr = instruction(op)
            skip_constant(instr) || push!(q.args, instr.instr)
        end
    end
end
function add_external_functions!(q::Expr, ls::LoopSet)
    for op ∈ operations(ls)
        if iscompute(op)
            instr = instruction(op)
            if instr.mod !== :LoopVectorization
                push!(q.args, instr.instr)
            end
        end
    end
end

function check_if_empty(ls::LoopSet, q::Expr)
    lb = loop_boundaries(ls, fill(false, length(ls.loops)))
    Expr(:if, Expr(:call, :!, Expr(:call, :any, :isempty, lb)), q)
end

val(x) = Expr(:call, Expr(:curly, :Val, x))

@inline gespf1(x, i) = gesp(x, i)
@inline gespf1(x::StridedPointer{T,1}, i::Tuple{I}) where {T,I<:Integer} = gesp(x, i)
@inline gespf1(x::StridedBitPointer{T,1}, i::Tuple{I}) where {T,I<:Integer} = gesp(x, i)
@inline gespf1(x::StridedPointer{T,1}, i::Tuple{Zero}) where {T} = x
@inline gespf1(x::StridedBitPointer{T,1}, i::Tuple{Zero}) where {T} = x
@generated function gespf1(x::StridedPointer{T,N,C,B,R}, i::Tuple{I}) where {T,N,I<:Integer,C,B,R}
  I === Zero && return :x
  ri = 0; rm = typemax(Int)
  for (i, r) ∈ enumerate(R)
    if r < rm
      rm = r
      ri = i
    end
  end
  ri = max(1, ri)
  quote
    $(Expr(:meta,:inline))
    p, li = VectorizationBase.tdot(x, (vsub_fast(getfield(i,1,false),1),), VectorizationBase.strides(x))
    ptr = gep(p, li)
    StridedPointer{$T,1,$(C===1 ? 1 : 0),$(B===1 ? 1 : 0),$(R[ri],)}(ptr, (getfield(getfield(x,:strd), $ri, 1),), (Zero(),))
  end
end
@generated function gespf1(x::StridedBitPointer{T,N,C,B,R}, i::Tuple{I}) where {T,N,I<:Integer,C,B,R}
  I === Zero && return :x
  quote
    $(Expr(:meta,:inline))
    p, li = VectorizationBase.tdot(x, (vsub_fast(getfield(i,1,false),1),), VectorizationBase.strides(x))
    ptr = gep(p, li)
    StridedBitPointer{1,$(C===1 ? 1 : 0),$(B===1 ? 1 : 0),$(first(R),)}(ptr, (first(getfield(x,:strd)),), (Zero(),))
  end
end
function findfirstcontaining(ref, ind)
  for (i,indr) ∈ enumerate(getindices(ref))
    ind === indr && return i
  end
  0
end
function should_zerorangestart(ls::LoopSet, allarrayrefs::Vector{ArrayReferenceMeta}, name_to_array_map::Vector{Vector{Int}})
  loops = ls.loops
  shouldindbyind = fill(false, length(loops))
  for (i,loop) ∈ enumerate(loops)
    ind = loop.itersymbol
    if isloopvalue(ls, ind)
      # we don't zero the range if it is used as a loopvalue
      shouldindbyind[i] = true
      continue
    end
    # otherwise, we need
    for namev ∈ name_to_array_map
      # firstcontainsind relies on stripping of duplicate inds in parsing
      firstcontainsind = findfirstcontaining(allarrayrefs[first(namev)], ind)
      allsame = true
      # The idea here is that if any ref to the same array doesn't have `ind`,
      # we can't offset that dimension because different inds will clash.
      # Because offseting the array means counter-offseting the range, we need
      # to be consistent, and check that all arrays are valid first.
      for j ∈ @view(namev[2:end])
        ref = allarrayrefs[j]
        if firstcontainsind ≠ findfirstcontaining(allarrayrefs[j], ind)
          allsame = false
          break
        end
      end
      if !allsame
        shouldindbyind[i] = true
        break
      end
    end
  end
  return shouldindbyind
end
function check_shouldindbyind(ls::LoopSet, ind::Symbol, shouldindbyind::Vector{Bool})
  for (i,loop) ∈ enumerate(ls.loops)
    loop.itersymbol === ind && return shouldindbyind[i]
  end
  true
end
# write a "check_loops_safe_to_zerorangestart
# that will be used to
# 1) decide whether to zerorangestart
# 2) decide whether to gesp that loopstart inside `add_grouped_strided_pointer`
function add_grouped_strided_pointer!(extra_args::Expr, ls::LoopSet)
  allarrayrefs, name_to_array_map, unique_to_name_and_op_map = uniquearrayrefs_csesummary(ls)
  shouldindbyind = should_zerorangestart(ls, allarrayrefs, name_to_array_map)
  # @show allarrayrefs
  gsp = Expr(:call, lv(:grouped_strided_pointer))
  tgarrays = Expr(:tuple)
  i = 0
  preserve_assignment = Expr(:tuple); preserve = Symbol[];
  @unpack equalarraydims, refs_aliasing_syms = ls
    # duplicate_map = collect(1:length(refs_aliasing_syms))
  duplicate_map = Vector{Int}(undef, length(refs_aliasing_syms))

  # for (i,j) ∈ enumerate(array_refs_with_same_name) # iterate over unique names
  #   ar = allarrayrefs[j]
  #   gespinds = cse_constant_offsets!(ls, allarrayrefs, j, array_refs_with_same_name, arrayref_to_name_op_collection)
  # end
  # @show refs_aliasing_syms
  for (j,ref) ∈ enumerate(refs_aliasing_syms)
    vpref = vptr(ref)
    duplicate = false
    for k ∈ 1:j-1 # quadratic, but should be short enough so that this is faster than O(1) algs
      if vptr(refs_aliasing_syms[k]) === vpref
        duplicate = true
        break
      end
    end
    # @show duplicate
    duplicate && continue
    duplicate_map[j] = (i += 1)
    found = false
    for j ∈ eachindex(allarrayrefs)
      if sameref(allarrayrefs[j], ref)
        gespinds = cse_constant_offsets!(ls, allarrayrefs, j, name_to_array_map, unique_to_name_and_op_map, shouldindbyind)
        push!(tgarrays.args, Expr(:call, lv(:gespf1), vpref, gespinds))
        found = true
        break
      end
    end
    @assert found
    push!(preserve, presbufsym(ref.ref.array))
  end
  push!(gsp.args, tgarrays)
  matcheddims = Expr(:tuple)
  for (vptrs,dims) ∈ equalarraydims
    t = Expr(:tuple)
    for (vp,d) ∈ zip(vptrs,dims)
      _id = findfirst(Base.Fix2(===,vp) ∘ vptr, refs_aliasing_syms)
      _id === nothing && continue
      push!(t.args, Expr(:tuple, duplicate_map[_id], d))
    end
    length(t.args) > 1 && push!(matcheddims.args, t)
  end
  push!(gsp.args, val(matcheddims))
  gsps = gensym!(ls, "#grouped#strided#pointer#")
  push!(extra_args.args, gsps)
  pushpreamble!(ls, Expr(:(=), gsps, Expr(:call, GlobalRef(Core,:getfield), gsp, 1)))
  preserve, shouldindbyind
end

# first_cache() = ifelse(gt(num_cache_levels(), StaticInt{2}()), StaticInt{2}(), StaticInt{1}())
# function _first_cache_size(::StaticInt{FCS}) where {FCS}
#     L1inclusive = StaticInt{FCS}() - VectorizationBase.cache_size(One())
#     ifelse(eq(first_cache(), StaticInt(2)) & VectorizationBase.cache_inclusive(StaticInt(2)), L1inclusive, StaticInt{FCS}())
# end
# _first_cache_size(::Nothing) = StaticInt(262144)
# first_cache_size() = _first_cache_size(cache_size(first_cache()))

@generated function _avx_config_val(
    ::Val{CNFARG}, ::StaticInt{W}, ::StaticInt{RS}, ::StaticInt{AR}, ::StaticInt{NT},
    ::StaticInt{CLS}, ::StaticInt{L1}, ::StaticInt{L2}, ::StaticInt{L3}
) where {CNFARG,W,RS,AR,CLS,L1,L2,L3,NT}
    inline,u₁,u₂,BROADCAST,thread = CNFARG
    nt = min(thread % UInt, NT % UInt)
    t = Expr(:tuple, inline, u₁, u₂, BROADCAST, W, RS, AR, CLS, L1,L2,L3, nt)
    Expr(:call, Expr(:curly, :Val, t))
end
@inline function avx_config_val(
    ::Val{CNFARG}, ::StaticInt{W}
) where {CNFARG,W}
    _avx_config_val(
        Val{CNFARG}(), StaticInt{W}(), register_size(), available_registers(), lv_max_num_threads(),
        cache_linesize(), cache_size(StaticInt(1)), cache_size(StaticInt(2)), cache_size(StaticInt(3))
    )
end
function find_samename_constparent(op::Operation, opname::Symbol)
    for opp ∈ parents(op)
        (((isconstant(opp) && instruction(opp) == LOOPCONSTANT) && (name(opp) === opname))) && return opp
        opptemp = find_samename_constparent(opp, opname)
        opptemp === opp || return opptemp
    end
    op
end
function remove_outer_reducts!(roots::Vector{Bool}, ls::LoopSet)
    ops = operations(ls)
    for or ∈ ls.outer_reductions
        op = ops[or]
        optemp = find_samename_constparent(op, name(op))
        if optemp ≢ op
            roots[identifier(optemp)] = false
        end
    end
end

# Try to condense in type stable manner
function generate_call(ls::LoopSet, (inline,u₁,u₂)::Tuple{Bool,Int8,Int8}, thread::UInt, debug::Bool = false)
    extra_args = Expr(:tuple)
    preserve, shouldindbyind = add_grouped_strided_pointer!(extra_args, ls)

    operation_descriptions = Expr(:tuple)
    varnames = Symbol[]; ids = Vector{Int}(undef, length(operations(ls)))
    ops = operations(ls)
    roots = getroots(ls)
    length(ls.includedactualarrays) == 0 || remove_outer_reducts!(roots, ls)
    for op ∈ ops
        instr::Instruction = instruction(op)
        if (isconstant(op) && (instr == LOOPCONSTANT)) && (!roots[identifier(op)])
            instr = op.instruction = DROPPEDCONSTANT
        end
        push!(operation_descriptions.args, QuoteNode(instr.mod))
        push!(operation_descriptions.args, QuoteNode(instr.instr))
        push!(operation_descriptions.args, OperationStruct!(varnames, ids, ls, op))
    end
    arraysymbolinds = Symbol[]
    arrayref_descriptions = Expr(:tuple)
    duplicate_ref = fill(false, length(ls.refs_aliasing_syms))
    for (j,ref) ∈ enumerate(ls.refs_aliasing_syms)
        vpref = vptr(ref)
        # duplicate_ref[j] ≠ 0 && continue
        duplicate_ref[j] && continue
        push!(arrayref_descriptions.args, ArrayRefStruct(ls, ref, arraysymbolinds, ids))
    end
    argmeta = argmeta_and_consts_description(ls, arraysymbolinds)
    loop_bounds = loop_boundaries(ls, shouldindbyind)
    loop_syms = tuple_expr(QuoteNode, ls.loopsymbols)
    func = debug ? lv(:_avx_loopset_debug) : lv(:_avx_!)
    lbarg = debug ? Expr(:call, :typeof, loop_bounds) : loop_bounds
    configarg = (inline,u₁,u₂,ls.isbroadcast,thread)
    unroll_param_tup = Expr(:call, lv(:avx_config_val), :(Val{$configarg}()), VECTORWIDTHSYMBOL)
    q = Expr(:call, func, unroll_param_tup, val(operation_descriptions), val(arrayref_descriptions), val(argmeta), val(loop_syms))
    
    add_reassigned_syms!(extra_args, ls) # counterpart to `add_ops!` constants
    for (opid,sym) ∈ ls.preamble_symsym # counterpart to process_metadata! symsym extraction
        if instruction(ops[opid]) ≠ DROPPEDCONSTANT
            push!(extra_args.args, sym)
        end
    end
    append!(extra_args.args, arraysymbolinds) # add_array_symbols!
    add_external_functions!(extra_args, ls) # extract_external_functions!
    push!(q.args, Expr(:tuple, lbarg, extra_args))
    vecwidthdefq = Expr(:block)
    define_eltype_vec_width!(vecwidthdefq, ls, nothing)
    Expr(:block, vecwidthdefq, q), preserve
end


"""
    check_args(::Vararg{AbstractArray})


LoopVectorization will optimize an `@avx` loop if `check_args` on each on the indexed abstract arrays returns true.
It returns true for `AbstractArray{T}`s when `check_type(T) == true` and the array or its parent is a `StridedArray` or `AbstractRange`.

To provide support for a custom array type, ensure that `check_args` returns true, either through overloading it or subtyping `DenseArray`.
Additionally, define `pointer` and `stride` methods.
"""
@inline function check_args(A::AbstractArray{T}) where {T}
    check_type(T) && check_device(ArrayInterface.device(A))
end
@inline check_args(A::BitVector) = true
@inline check_args(A::BitArray) = iszero(size(A,1) & 7)
@inline check_args(::VectorizationBase.AbstractStridedPointer) = true
@inline function check_args(x)
    # @info "`LoopVectorization.check_args(::$(typeof(x))) == false`, therefore compiling a probably slow `@inbounds @fastmath` fallback loop." maxlog=1
    false
end
@inline check_args(A, B, C::Vararg{Any,K}) where {K} = check_args(A) && check_args(B, C...)
@inline check_args(::AbstractRange{T}) where {T} = check_type(T)
@inline check_args(::UpTri) = false
@inline check_args(::LoTri) = false
@inline check_args(::Diagonal) = false
@inline check_args(::Type{T}) where {T} = check_type(T)
"""
    check_type(::Type{T}) where {T}

Returns true if the element type is supported.
"""
@inline check_type(::Type{T}) where {T <: NativeTypes} = true
@inline check_type(::Type{T}) where {T} = false
@inline check_device(::ArrayInterface.CPUPointer) = true
@inline check_device(::ArrayInterface.CPUTuple) = true
@inline check_device(x) = false

function check_args_call(ls::LoopSet)
    q = Expr(:call, lv(:check_args))
    append!(q.args, ls.includedactualarrays)
    for r ∈ ls.outer_reductions
        push!(q.args, Expr(:call, :typeof, name(ls.operations[r])))
    end
    q
end

make_fast(q) = Expr(:macrocall, Symbol("@fastmath"), LineNumberNode(@__LINE__,Symbol(@__FILE__)), q)
make_crashy(q) = Expr(:macrocall, Symbol("@inbounds"), LineNumberNode(@__LINE__,Symbol(@__FILE__)), q)

@inline vecmemaybe(x::NativeTypes) = x
@inline vecmemaybe(x::VectorizationBase._Vec) = Vec(x)
@inline vecmemaybe(x::Tuple) = VectorizationBase.VecUnroll(x)
@inline vecmemaybe(x::Mask) = x

function gc_preserve(call::Expr, preserve::Vector{Symbol})
    q = Expr(:gc_preserve, call)
    append!(q.args, preserve)
    q
end

function setup_call_inline(ls::LoopSet, inline::Bool, u₁::Int8, u₂::Int8, thread::Int)
    call, preserve = generate_call(ls, (inline,u₁,u₂), thread % UInt, false)
    if iszero(length(ls.outer_reductions))
        pushpreamble!(ls, gc_preserve(call, preserve))
        push!(ls.preamble.args, nothing)
        return ls.preamble
    end
    retv = loopset_return_value(ls, Val(false))
    outer_reducts = Expr(:local)
    q = Expr(:block,gc_preserve(Expr(:(=), retv, call), preserve))
    for or ∈ ls.outer_reductions
        op = ls.operations[or]
        var = name(op)
        # push!(call.args, Symbol("##TYPEOF##", var))
        mvar = mangledvar(op)
        instr = instruction(op)
        out = Symbol(mvar, "##onevec##")
        push!(outer_reducts.args, out)
        push!(q.args, Expr(:(=), var, Expr(:call, lv(reduction_scalar_combine(instr)), Expr(:call, lv(:vecmemaybe), out), var)))
    end
    pushpreamble!(ls, outer_reducts)
    append!(ls.preamble.args, q.args)
    ls.preamble
end
function setup_call_debug(ls::LoopSet)
    # avx_loopset(instr, ops, arf, AM, LB, vargs)
    pushpreamble!(ls, first(generate_call(ls, (false,zero(Int8),zero(Int8)), zero(UInt), true)))
    Expr(:block, ls.prepreamble, ls.preamble)
end
function setup_call(
    ls::LoopSet, q::Expr, source::LineNumberNode, inline::Bool, check_empty::Bool, u₁::Int8, u₂::Int8, thread::Int
)
    # We outline/inline at the macro level by creating/not creating an anonymous function.
    # The old API instead was based on inlining or not inline the generated function, but
    # the generated function must be inlined into the initial loop preamble for performance reasons.
    # Creating an anonymous function and calling it also achieves the outlining, while still
    # inlining the generated function into the loop preamble.
    lnns = extract_all_lnns(q)
    pushfirst!(lnns, source)
    call = setup_call_inline(ls, inline, u₁, u₂, thread)
    call = check_empty ? check_if_empty(ls, call) : call
    result = Expr(:block, ls.prepreamble, Expr(:if, check_args_call(ls), call, make_crashy(make_fast(q))))
    prepend_lnns!(result, lnns)
    return result
end
