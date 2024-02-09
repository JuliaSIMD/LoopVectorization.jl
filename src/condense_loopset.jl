@enum IndexType::UInt8 NotAnIndex = 0 LoopIndex = 1 ComputedIndex = 2 SymbolicIndex =
  3

Base.:|(u::Unsigned, it::IndexType) = u | UInt8(it)
Base.:(==)(u::Unsigned, it::IndexType) = (u % UInt8) == UInt8(it)

"""
    ArrayRefStruct

A condensed representation of an [`ArrayReference`](@ref).
It supports array-references with up to 8 indexes, where the data for each consecutive index is packed into corresponding 8-bit fields
of `index_types` (storing the enum `IndexType`), `indices` (the `id` for each index symbol), and `offsets` (currently unused).
"""
struct ArrayRefStruct{array,ptr}
  index_types::UInt128
  indices::UInt128
  offsets::UInt128
  strides::UInt128
end
array_and_ptr(@nospecialize(ar::ArrayRefStruct{a,p})) where {a,p} =
  (a::Symbol, p::Symbol)
# array(@nospecialize(ar::ArrayRefStruct{a,p})) where {a,p} = a::Symbol
# ptr(@nospecialize(ar::ArrayRefStruct{a,p})) where {a,p}   = p::Symbol

function findindoradd!(v::Vector{T}, s::T) where {T}
  ind = findfirst(==(s), v)
  ind === nothing || return ind
  push!(v, s)
  length(v)
end
function ArrayRefStruct(
  ls::LoopSet,
  mref::ArrayReferenceMeta,
  arraysymbolinds::Vector{Symbol},
  ids::Vector{Int}
)
  index_types = zero(UInt128)
  indices = zero(UInt128)
  offsets = zero(UInt128)
  strides = zero(UInt128)
  @unpack loopedindex, ref = mref
  indv = ref.indices
  offv = ref.offsets
  strv = ref.strides
  # we can discard that the array was considered discontiguous, as it should be recovered from type information
  start = 1 + (first(indv) === DISCONTIGUOUS)
  for (n, ind) ∈ enumerate(@view(indv[start:end]))
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
  ArrayRefStruct{mref.ref.array,mref.ptr}(
    index_types,
    indices,
    offsets,
    strides
  )
end

"""
    OperationStruct

A condensed representation of an [`Operation`](@ref).
"""
struct OperationStruct <: AbstractLoopOperation
  # instruction::Instruction
  loopdeps::UInt128
  reduceddeps::UInt128
  childdeps::UInt128
  parents₀::UInt128
  parents₁::UInt128
  parents₂::UInt128
  parents₃::UInt128
  node_type::OperationType
  symid::UInt16
  array::UInt16
end
optype(os) = os.node_type

function findmatchingarray(ls::LoopSet, mref::ArrayReferenceMeta)
  id = 0x0001
  for r ∈ ls.refs_aliasing_syms
    r == mref && return id
    id += 0x0001
  end
  0x0000
end
filled_8byte_chunks(u::T) where {T<:Unsigned} =
  sizeof(T) - (leading_zeros(u) >>> 3)

function shifted_loopset(ls::LoopSet, loopsyms::Vector{Symbol})
  ld = zero(UInt128) # leading_zeros(ld) >> 2 yields the number of loopdeps
  for d ∈ loopsyms
    ld <<= 4
    ld |= getloopid(ls, d)::Int
  end
  ld
end
loopdeps_uint(ls::LoopSet, op::Operation) =
  shifted_loopset(ls, loopdependencies(op))
reduceddeps_uint(ls::LoopSet, op::Operation) =
  shifted_loopset(ls, reduceddependencies(op))
childdeps_uint(ls::LoopSet, op::Operation) =
  shifted_loopset(ls, reducedchildren(op))
function parents_uint(oppv::AbstractVector{Operation})
  p = zero(UInt128)
  for parent ∈ oppv
    p <<= 16
    p |= identifier(parent)
  end
  p
end
function parents_uint(op::Operation)
  opv = parents(op)
  N = length(opv)
  @assert N ≤ 32
  p0 = parents_uint(view(opv, 1:min(8, N)))
  p1 = N > 8 ? parents_uint(view(opv, 9:min(16, N))) : zero(p0)
  p2 = N > 16 ? parents_uint(view(opv, 17:min(24, N))) : zero(p0)
  p3 = N > 24 ? parents_uint(view(opv, 25:N)) : zero(p0)
  p0, p1, p2, p3
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
  remove_outer_reducts!(rooted, ls)
  return rooted
end
function OperationStruct!(
  varnames::Vector{Symbol},
  ids::Vector{Int},
  ls::LoopSet,
  op::Operation
)
  ld = loopdeps_uint(ls, op)
  rd = reduceddeps_uint(ls, op)
  cd = childdeps_uint(ls, op)
  p0, p1, p2, p3 = parents_uint(op)
  array = accesses_memory(op) ? findmatchingarray(ls, op.ref) : 0x0000
  ids[identifier(op)] = id = findindoradd!(varnames, name(op))
  OperationStruct(ld, rd, cd, p0, p1, p2, p3, op.node_type, id, array)
end
## turn a LoopSet into a type object which can be used to reconstruct the LoopSet.

@inline zerorangestart(r::Base.OneTo) = CloseOpen(maybestaticlast(r))
@inline zerorangestart(r::AbstractCloseOpen) = CloseOpen(length(r))
@inline zerorangestart(r::AbstractCloseOpen{Zero}) = r
@inline zerorangestart(r::AbstractUnitRange) =
  Zero():One():(maybestaticlast(r)-maybestaticfirst(r))
@inline zerorangestart(r::AbstractRange) =
  Zero():static_step(r):(maybestaticlast(r)-maybestaticfirst(r))
@inline zerorangestart(r::CartesianIndices) =
  CartesianIndices(map(zerorangestart, r.indices))
@inline zerorangestart(
  r::ArrayInterface.OptionallyStaticUnitRange{StaticInt{1}}
) = CloseOpen(maybestaticlast(r))

function loop_boundary!(q::Expr, loop::Loop, shouldindbyind::Bool)
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
    loop_boundary!(lbd, loop, ibi)
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
        push!(
          ret.args,
          Expr(:call, :vdata, Symbol(mangledvar(op), "##onevec##"))
        )
      else
        push!(ret.args, Symbol(mangledvar(ops[or]), "##onevec##"))
      end
    end
    ret
  end
end
const DROPPEDCONSTANT =
  Instruction(Symbol("##DROPPED#CONSTANT##"), Symbol("##DROPPED#CONSTANT##"))
function skip_constant(instr::Instruction)
  (
    ((instr == LOOPCONSTANT) || (instr.mod === :numericconstant)) ||
    (instr == DROPPEDCONSTANT)
  ) || instr.mod === GLOBALCONSTANT
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
@inline gespf1(
  x::StridedPointer{T,1},
  i::Tuple{I}
) where {T,I<:Union{Integer,StaticInt}} = gesp(x, i)
@inline gespf1(
  x::StridedBitPointer{T,1},
  i::Tuple{I}
) where {T,I<:Union{Integer,StaticInt}} = gesp(x, i)
@inline gespf1(x::StridedPointer{T,1}, i::Tuple{Zero}) where {T} = x
@inline gespf1(x::StridedBitPointer{T,1}, i::Tuple{Zero}) where {T} = x
@generated function gespf1(
  x::AbstractStridedPointer{T,N,C,B,R},
  i::Tuple{I}
) where {T,N,I<:Union{Integer,StaticInt},C,B,R}
  ri = argmin(R)
  quote
    $(Expr(:meta, :inline))
    p, li = VectorizationBase.tdot(
      x,
      (vsub_nsw(getfield(i, 1), one($I)),),
      static_strides(x)
    )
    ptr = gep(p, li)
    si = ArrayInterface.StrideIndex{1,$(R[ri],),$(C === 1 ? 1 : 0)}(
      (getfield(static_strides(x), $ri),),
      (Zero(),)
    )
    stridedpointer(ptr, si, StaticInt{$(B === 1 ? 1 : 0)}())
  end
end
@generated function gespf1(
  x::AbstractStridedPointer{T,N,C,B,R},
  ::Tuple{VectorizationBase.NullStep}
) where {T,N,C,B,R}
  ri = argmin(R)
  quote
    $(Expr(:meta, :inline))
    si = ArrayInterface.StrideIndex{1,$(R[ri],),$(C === 1 ? 1 : 0)}(
      (getfield(static_strides(x), $ri),),
      (getfield(offsets(x), $ri),)
    )
    stridedpointer(pointer(x), si, StaticInt{$(B == 1 ? 1 : 0)}())
  end
end
function findfirstcontaining(ref, ind)
  for (i, indr) ∈ enumerate(getindicesonly(ref))
    ind === indr && return i
  end
  0
end
function should_zerorangestart(
  ls::LoopSet,
  allarrayrefs::Vector{ArrayReferenceMeta},
  name_to_array_map::Vector{Vector{Int}},
  isrooted::Vector{Bool}
)
  loops = ls.loops
  shouldindbyind = fill(false, length(loops))
  for (i, loop) ∈ enumerate(loops)
    ind = loop.itersymbol
    if isloopvalue(ls, ind, isrooted)
      # we don't zero the range if it is used as a loopvalue
      shouldindbyind[i] = true
      continue
    end
    # otherwise, we need
    for namev ∈ name_to_array_map
      baseref = allarrayrefs[first(namev)]
      # firstcontainsind relies on stripping of duplicate inds in parsing
      firstcontainsind = findfirstcontaining(baseref, ind)
      basestride =
        firstcontainsind == 0 ? 0 : getstrides(baseref)[firstcontainsind]
      allsame = true
      # The idea here is that if any ref to the same array doesn't have `ind`,
      # we can't offset that dimension because different inds will clash.
      # Because offsetting the array means counter-offsetting the range, we need
      # to be consistent, and check that all arrays are valid first.
      for j ∈ @view(namev[2:end])
        ref = allarrayrefs[j]
        if (firstcontainsind ≠ findfirstcontaining(ref, ind)) || (
          (firstcontainsind ≠ 0) &&
          (basestride ≠ getstrides(ref)[firstcontainsind])
        )
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
function check_shouldindbyind(
  ls::LoopSet,
  ind::Symbol,
  shouldindbyind::Vector{Bool}
)
  for (i, loop) ∈ enumerate(ls.loops)
    loop.itersymbol === ind && return shouldindbyind[i]
  end
  true
end

@inline densewrapper(sp, A) = sp
@inline densewrapper(
  sp::AbstractStridedPointer{T,N},
  A::AbstractArray{T,N}
) where {T,N} = _densewrapper(sp, VectorizationBase.val_dense_dims(A))
@inline _densewrapper(sp, ::Nothing) = sp
@inline _densewrapper(sp::AbstractStridedPointer, ::Val{D}) where {D} =
  VectorizationBase.DensePointerWrapper{D}(sp)

# write a "check_loops_safe_to_zerorangestart
# that will be used to
# 1) decide whether to zerorangestart
# 2) decide whether to gesp that loopstart inside `add_grouped_strided_pointer`
function add_grouped_strided_pointer!(extra_args::Expr, ls::LoopSet)
  allarrayrefs, name_to_array_map, unique_to_name_and_op_map =
    uniquearrayrefs_csesummary(ls)
  gsp = Expr(:call, lv(:grouped_strided_pointer))
  tgarrays = Expr(:tuple)
  # refs_to_gesp = ArrayReferenceMeta[]
  gespsummaries = Tuple{Int,Vector{Symbol}}[]
  i = 0
  preserve_assignment = Expr(:tuple)
  preserve = Symbol[]
  @unpack equalarraydims, refs_aliasing_syms = ls
  # duplicate_map = collect(1:length(refs_aliasing_syms))
  duplicate_map = Vector{Int}(undef, length(refs_aliasing_syms))

  # for (i,j) ∈ enumerate(array_refs_with_same_name) # iterate over unique names
  #   ar = allarrayrefs[j]
  #   gespinds = cse_constant_offsets!(ls, allarrayrefs, j, array_refs_with_same_name, arrayref_to_name_op_collection)
  # end
  for (j, ref) ∈ enumerate(refs_aliasing_syms)
    vpref = vptr(ref)
    duplicate = false
    for k ∈ 1:j-1 # quadratic, but should be short enough so that this is faster than O(1) algs
      if vptr(refs_aliasing_syms[k]) === vpref
        duplicate = true
        break
      end
    end
    duplicate && continue
    duplicate_map[j] = (i += 1)
    found = false
    for k ∈ eachindex(allarrayrefs)
      if sameref(allarrayrefs[k], ref)
        gespindsummary = cse_constant_offsets!(
          ls,
          allarrayrefs,
          k,
          name_to_array_map,
          unique_to_name_and_op_map
        )
        push!(gespsummaries, (k, gespindsummary))
        found = true
        break
      end
    end
    @assert found
    push!(preserve, presbufsym(ref.ref.array))
  end
  roots = getroots(ls)
  shouldindbyind =
    should_zerorangestart(ls, allarrayrefs, name_to_array_map, roots)
  for (k, gespindsummary) ∈ gespsummaries
    ref = allarrayrefs[k]
    gespinds = calcgespinds(
      ls,
      ref,
      gespindsummary,
      shouldindbyind,
      name_to_array_map[first(first(unique_to_name_and_op_map[k]))],
      unique_to_name_and_op_map
    )
    push!(
      tgarrays.args,
      Expr(
        :call,
        lv(:densewrapper),
        Expr(:call, lv(:gespf1), vptr(ref), gespinds),
        name(ref)
      )
    )
  end
  push!(gsp.args, tgarrays)
  matcheddims = Expr(:tuple)
  for (vptrs, dims) ∈ equalarraydims
    t = Expr(:tuple)
    for (vp, d) ∈ zip(vptrs, dims)
      _id = findfirst(Base.Fix2(===, vp) ∘ vptr, refs_aliasing_syms)
      _id === nothing && continue
      push!(t.args, Expr(:tuple, duplicate_map[_id], d))
    end
    length(t.args) > 1 && push!(matcheddims.args, t)
  end
  push!(gsp.args, val(matcheddims))
  gsps = gensym!(ls, "#grouped#strided#pointer#")
  push!(extra_args.args, gsps)
  pushpreamble!(ls, Expr(:(=), gsps, Expr(:call, getfield, gsp, 1)))
  preserve, shouldindbyind, roots
end

@generated function _turbo_config_val(
  ::Val{CNFARG},
  ::StaticInt{W},
  ::StaticInt{RS},
  ::StaticInt{AR},
  ::StaticInt{NT},
  ::StaticInt{CLS}
) where {CNFARG,W,RS,AR,CLS,NT}
  inline, u₁, u₂, v, BROADCAST, thread, warncheckarg, safe = CNFARG
  nt = min(thread % UInt, NT % UInt)
  t = Expr(
    :tuple,
    inline,
    u₁,
    u₂,
    v,
    BROADCAST,
    W,
    RS,
    AR,
    CLS,
    nt,
    warncheckarg,
    safe
  )
  length(CNFARG) == 7 && push!(t.args, CNFARG[7])
  Expr(:call, Expr(:curly, :Val, t))
end
@inline function avx_config_val(::Val{CNFARG}, ::StaticInt{W}) where {CNFARG,W}
  _turbo_config_val(
    Val{CNFARG}(),
    StaticInt{W}(),
    register_size(),
    available_registers(),
    num_cores(), #FIXME
    cache_linesize()
  )
end
function find_samename_constparent(op::Operation, opname::Symbol)
  for opp ∈ parents(op)
    ((
      (isconstant(opp) && instruction(opp) == LOOPCONSTANT) &&
      (name(opp) === opname)
    )) && return opp
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

function split_ifelse!(
  ls::LoopSet,
  preserve::Vector{Symbol},
  shouldindbyind::Vector{Bool},
  roots::Vector{Bool},
  extra_args::Expr,
  k::Int,
  inlineu₁u₂::Tuple{Bool,Int8,Int8,Int8},
  thread::UInt,
  warncheckarg::Int,
  safe::Bool,
  debug::Bool
)
  roots[k] = false
  op = operations(ls)[k]
  op.instruction = DROPPEDCONSTANT
  op.node_type = constant
  # we want to eliminate
  parents_op = parents(op)
  condop = first(parents_op)
  # create one loop where `opp` is true, and a second where it is `false`
  prepre = ls.prepreamble
  append!(prepre.args, ls.preamble.args)
  ls.prepreamble = Expr(:block)
  ls.preamble = Expr(:block)
  ls_true = deepcopy(ls)
  lsfalse = ls
  true_ops = operations(ls_true)
  falseops = operations(lsfalse)
  true_op = parents(true_ops[k])[2]
  falseop = parents_op[3]
  true_op.dependencies = loopdependencies(op)
  falseop.dependencies = loopdependencies(op)
  true_op.reduced_children = reducedchildren(op)
  falseop.reduced_children = reducedchildren(op)
  condop_count = 0
  for i ∈ eachindex(falseops)
    fop = falseops[i]
    parents_false = parents(fop)
    for (j, opp) ∈ enumerate(parents_false)
      if opp === op # then ops[i]'s jth parent is the ifelse
        # These reduction to scalar instructions are added for non-outer reductions initialized with non-constant ops
        # So we check if now 
        # if (j == 2) && (Base.sym_in(instruction(fop).instr, (:reduced_add, :reduced_prod, :reduced_max, :reduced_min, :reduced_all, :reduced_any)))
        # if isconstantop(true_op)
        #   (true_ops[i]).instruction = Instruction(:identity)
        # end
        # if isconstantop(falseop)
        #   fop.instruction = Instruction(:identity)
        # end
        # end
        parents(true_ops[i])[j] = true_op
        parents_false[j] = falseop
      end
      condop_count += roots[i] & (condop === opp)
    end
  end
  roots[identifier(condop)] &= condop_count > 0
  q = :(
    if $(name(condop))
      $(generate_call_split(
        ls_true,
        preserve,
        shouldindbyind,
        roots,
        copy(extra_args),
        inlineu₁u₂,
        thread,
        warncheckarg,
        safe,
        debug
      ))
    else
      $(generate_call_split(
        lsfalse,
        preserve,
        shouldindbyind,
        roots,
        extra_args,
        inlineu₁u₂,
        thread,
        warncheckarg,
        safe,
        debug
      ))
    end
  )
  push!(prepre.args, q)
  prepre
end

function generate_call(
  ls::LoopSet,
  inlineu₁u₂::Tuple{Bool,Int8,Int8,Int8},
  thread::UInt,
  warncheckarg::Int,
  safe::Bool,
  debug::Bool
)
  extra_args = Expr(:tuple)
  fill_children!(ls)
  preserve, shouldindbyind, roots = add_grouped_strided_pointer!(extra_args, ls)
  generate_call_split(
    ls,
    preserve,
    shouldindbyind,
    roots,
    extra_args,
    inlineu₁u₂,
    thread,
    warncheckarg,
    safe,
    debug
  )
end
function generate_call_split(
  ls::LoopSet,
  preserve::Vector{Symbol},
  shouldindbyind::Vector{Bool},
  roots::Vector{Bool},
  extra_args::Expr,
  inlineu₁u₂::Tuple{Bool,Int8,Int8,Int8},
  thread::UInt,
  warncheckarg::Int,
  safe::Bool,
  debug::Bool
)
  for (k, op) ∈ enumerate(operations(ls))
    parents_op = parents(op)
    if (iscompute(op) && (instruction(op).instr === :ifelse)) &&
       (length(parents_op) == 3) &&
       isconstantop(first(parents_op))
      return split_ifelse!(
        ls,
        preserve,
        shouldindbyind,
        roots,
        extra_args,
        k,
        inlineu₁u₂,
        thread,
        warncheckarg,
        safe,
        debug
      )
    end
  end
  return generate_call_types(
    ls,
    preserve,
    shouldindbyind,
    roots,
    extra_args,
    inlineu₁u₂,
    thread,
    warncheckarg,
    safe,
    debug
  )
end

# Try to condense in type stable manner
function generate_call_types(
  ls::LoopSet,
  preserve::Vector{Symbol},
  shouldindbyind::Vector{Bool},
  roots::Vector{Bool},
  extra_args::Expr,
  (inline, u₁, u₂, v)::Tuple{Bool,Int8,Int8,Int8},
  thread::UInt,
  warncheckarg::Int,
  safe::Bool,
  debug::Bool
)
  # good place to check for split  
  operation_descriptions = Expr(:tuple)
  varnames = Symbol[]
  ids = Vector{Int}(undef, length(operations(ls)))
  ops = operations(ls)
  for op ∈ ops
    instr::Instruction = instruction(op)
    if (!roots[identifier(op)])
      if (isconstant(op) && (instr == LOOPCONSTANT)) || !isconstant(op)
        instr = op.instruction = DROPPEDCONSTANT
        op.node_type = constant
      end
    end
    push!(operation_descriptions.args, QuoteNode(instr.mod))
    push!(operation_descriptions.args, QuoteNode(instr.instr))
    push!(operation_descriptions.args, OperationStruct!(varnames, ids, ls, op))
  end
  arraysymbolinds = Symbol[]
  arrayref_descriptions = Expr(:tuple)
  duplicate_ref = fill(false, length(ls.refs_aliasing_syms))
  for (j, ref) ∈ enumerate(ls.refs_aliasing_syms)
    # duplicate_ref[j] ≠ 0 && continue
    duplicate_ref[j] && continue
    push!(
      arrayref_descriptions.args,
      ArrayRefStruct(ls, ref, arraysymbolinds, ids)
    )
  end
  argmeta = argmeta_and_consts_description(ls, arraysymbolinds)
  loop_bounds = loop_boundaries(ls, shouldindbyind)
  loop_syms = tuple_expr(QuoteNode, ls.loopsymbols)
  lbarg = debug ? Expr(:call, :typeof, loop_bounds) : loop_bounds
  configarg = (inline, u₁, u₂, v, ls.isbroadcast, thread, warncheckarg, safe)
  unroll_param_tup =
    Expr(:call, lv(:avx_config_val), :(Val{$configarg}()), VECTORWIDTHSYMBOL)
  add_reassigned_syms!(extra_args, ls) # counterpart to `add_ops!` constants
  for (opid, sym) ∈ ls.preamble_symsym # counterpart to process_metadata! symsym extraction
    if instruction(ops[opid]) ≠ DROPPEDCONSTANT
      push!(extra_args.args, sym)
    end
  end
  append!(extra_args.args, arraysymbolinds) # add_array_symbols!
  add_external_functions!(extra_args, ls) # extract_external_functions!
  add_outerreduct_types!(extra_args, ls) # extract_outerreduct_types!
  argcestimate = length(extra_args.args) - 1
  for ref in ls.refs_aliasing_syms
    argcestimate += length(ref.loopedindex)
  end
  manyarg = !debug && (argcestimate > 16)
  func =
    debug ? lv(:_turbo_loopset_debug) :
    (manyarg ? lv(:_turbo_manyarg!) : lv(:_turbo_!))
  q = Expr(
    :call,
    func,
    unroll_param_tup,
    val(operation_descriptions),
    val(arrayref_descriptions),
    val(argmeta),
    val(loop_syms)
  )
  vecwidthdefq = if debug
    push!(q.args, Expr(:tuple, lbarg, extra_args))
    Expr(:block)
  else
    vargsym = gensym(:vargsym)
    push!(
      q.args,
      Expr(
        :call,
        GlobalRef(Base, :Val),
        Expr(:call, GlobalRef(Base, :typeof), vargsym)
      )
    )
    if manyarg
      push!(q.args, vargsym)
    else
      push!(q.args, Expr(:(...), vargsym))
    end
    Expr(:block, Expr(:(=), vargsym, Expr(:tuple, lbarg, extra_args)))
  end
  define_eltype_vec_width!(vecwidthdefq, ls, nothing, true)
  push!(vecwidthdefq.args, q)
  if debug
    pushpreamble!(ls, vecwidthdefq)
    Expr(:block, ls.prepreamble, ls.preamble)
  else
    setup_call_final(ls, setup_outerreduct_preserve(ls, vecwidthdefq, preserve))
  end
end
# @inline reductinittype(::T) where {T} = StaticType{T}()
typeof_expr(op::Operation) = Expr(:call, GlobalRef(Base, :typeof), name(op))
eltype_expr(op::Operation) = Expr(:call, GlobalRef(Base, :eltype), name(op))
function add_outerreduct_types!(extra_args::Expr, ls::LoopSet) # extract_outerreduct_types!
  for or ∈ ls.outer_reductions
    op = operations(ls)[or]
    if instruction(op).instr ≢ :ifelse
      push!(extra_args.args, eltype_expr(op))
    else
      push!(extra_args.args, name(op))
    end
  end
end
"""
    check_args(::Vararg{AbstractArray})

LoopVectorization will optimize an `@turbo` loop if `check_args` on each on the indexed abstract arrays returns true.
It returns true for `AbstractArray{T}`s when `check_type(T) == true` and the array or its parent is a `StridedArray` or `AbstractRange`.

To provide support for a custom array type, ensure that `check_args` returns true, either through overloading it or subtyping `DenseArray`.
Additionally, define `pointer` and `stride` methods.
"""
@inline function check_args(A::AbstractArray{T}) where {T}
  check_type(T) && check_device(ArrayInterface.device(A))
end
@inline check_args(A::BitVector) = true
@inline check_args(A::BitArray) = iszero(size(A, 1) & 7)
@inline check_args(::VectorizationBase.AbstractStridedPointer) = true
@inline function check_args(x)
  # @info "`LoopVectorization.check_args(::$(typeof(x))) == false`, therefore compiling a probably slow `@inbounds @fastmath` fallback loop." maxlog=1
  # DEBUG: @show @__LINE__, typeof(x)
  false
end
@inline check_args(A, B, C::Vararg{Any,K}) where {K} =
  check_args(A) && check_args(B, C...)
@inline check_args(::AbstractRange{T}) where {T} = check_type(T)
@inline check_args(::UpTri) = false
@inline check_args(::LoTri) = false
@inline check_args(::Diagonal) = false
@inline check_args(::Type{T}) where {T} = check_type(T)
@inline check_args(::Tuple{T,Vararg{T,K}}) where {T,K} = check_type(T)
"""
    check_type(::Type{T}) where {T}

Returns true if the element type is supported.
"""
@inline check_type(::Type{T}) where {T<:NativeTypes} = true
@inline function check_type(::Type{T}) where {T}
  # DEBUG: @show @__LINE__, T
  false
end
@inline check_type(::Type{T}) where {T<:AbstractSIMD} = true
@inline check_device(::ArrayInterface.CPUPointer) = true
@inline check_device(::ArrayInterface.CPUTuple) = true
@inline function check_device(x)
  # DEBUG: @show @__LINE__, typeof(x)
  false
end

function check_args_call(ls::LoopSet)
  q = Expr(:call, lv(:check_args))
  append!(q.args, ls.includedactualarrays)
  for r ∈ ls.outer_reductions
    push!(q.args, Expr(:call, :typeof, name(ls.operations[r])))
  end
  q
end
struct RetVec2Int end
(::RetVec2Int)(_) = Vec{2,Int}
"""
can_turbo(f::Function, ::Val{NARGS})

Check whether a given function with a specified number of arguments
can be used inside a `@turbo` loop.
"""
function can_turbo(f::F, ::Val{NARGS})::Bool where {F,NARGS}
  promoted_op = Base.promote_op(f, ntuple(RetVec2Int(), Val(NARGS))...)
  # DEBUG: promoted_op === Union{} && @show f, NARGS
  return promoted_op !== Union{}
end
can_turbo(::typeof(vfmaddsub), ::Val{3}) = true
can_turbo(::typeof(vfmsubadd), ::Val{3}) = true
can_turbo(::typeof(ifelse), ::Val{3}) = true
can_turbo(::typeof(!), ::Val{1}) = true
can_turbo(::typeof(^), ::Val{2}) = true
can_turbo(::typeof(Base.literal_pow), ::Val{3}) = true
can_turbo(::typeof(Base.FastMath.pow_fast), ::Val{2}) = true

for f ∈ (convert, reinterpret, trunc, unsafe_trunc, round, ceil, floor)
  @eval can_turbo(::typeof($f), ::Val{2}) = true
end

# @inline function _can_turbo(f::F, t::Vararg{Any,K}) where {F,K}
#   Base.promote_op(f, t...) !== Union{}
# end

"""
    check_turbo_safe(ls::LoopSet)

Returns an expression of the form `true && can_turbo(op1) && can_turbo(op2) && ...`
"""
function check_turbo_safe(ls::LoopSet)
  q = Expr(:&&, true)
  last = q
  for op in operations(ls)
    iscompute(op) || continue
    c = callexpr(op.instruction)
    nargs = length(parents(op))
    push!(c.args, Val(nargs))
    pushfirst!(c.args, can_turbo)
    new_last = Expr(:&&, c)
    push!(last.args, new_last)
    last = new_last
  end
  q
end

make_fast(q) = Expr(
  :macrocall,
  Symbol("@fastmath"),
  LineNumberNode(@__LINE__, Symbol(@__FILE__)),
  q
)
make_crashy(q) = Expr(
  :macrocall,
  Symbol("@inbounds"),
  LineNumberNode(@__LINE__, Symbol(@__FILE__)),
  q
)

@inline vecmemaybe(x::NativeTypes) = x
@inline vecmemaybe(x::VectorizationBase._Vec) = Vec(x)
@inline vecmemaybe(x::VectorizationBase.Vec) = x
@inline vecmemaybe(x::Tuple) = VectorizationBase.VecUnroll(x)
@inline vecmemaybe(x::Mask) = x

function gc_preserve(call::Expr, preserve::Vector{Symbol})
  q = Expr(:gc_preserve, call)
  append!(q.args, preserve)
  q
end

# @generated function ifelse_reduce(f::F, x::Vec{W,T}) where {F,T,W}
#   Wt = W >>> 1
#   # uw = Symbol(:x_, Wt)
#   # lw = Symbol(:x_, Wt)
#   xw = Symbol(:x_, Wt)
#   q = Expr(:block, Expr(:meta,:inline), :($xw = f($(VectorizationBase.uppervector)(x), $(VectorizationBase.lowervector)(x))))
#   while Wt > 2
#     Wt >>>= 1
#     ex = :(f($(VectorizationBase.uppervector)($xw), $(VectorizationBase.lowervector)($xw)))
#     xw = Symbol(:x_, Wt)
#     push!(q.args, :($xw = $ex))
#   end

# end

# function setup_call_inline(ls::LoopSet, inline::Bool, u₁::Int8, u₂::Int8, thread::Int)
#   call, preserve = generate_call_split(ls, (inline,u₁,u₂), thread % UInt, false)
#   setup_call_ret!(ls, call, preserve)
# end
setup_outerreduct_preserve_mangler(op::Operation) =
  Symbol(mangledvar(op), "##onevec##")

function outer_reduction_to_scalar_reduceq!(
  q::Expr,
  op::Operation,
  var = name(op)
)
  instr = instruction(op)
  out = setup_outerreduct_preserve_mangler(op)
  if instr.instr ≢ :ifelse
    Expr(
      :call,
      reduction_scalar_combine(op),
      Expr(:call, lv(:vecmemaybe), out),
      var
    )
  else
    opinstr = ifelse_reduction(:IfElseReduced, op) do opv
      opvname = name(opv)
      oporig = gensym(opvname)
      pushfirst!(q.args, Expr(:(=), oporig, opvname))
      Expr(:call, lv(:vecmemaybe), setup_outerreduct_preserve_mangler(opv)),
      (oporig,)
    end
    Expr(:call, opinstr, Expr(:call, lv(:vecmemaybe), out), var)
  end
end
function setup_outerreduct_preserve(
  ls::LoopSet,
  call::Expr,
  preserve::Vector{Symbol}
)
  iszero(length(ls.outer_reductions)) && return gc_preserve(call, preserve)
  retv = loopset_return_value(ls, Val(false))
  q = Expr(:block, gc_preserve(Expr(:(=), retv, call), preserve))
  for or ∈ ls.outer_reductions
    op = ls.operations[or]
    # push!(call.args, Symbol("##TYPEOF##", var))
    reducq = outer_reduction_to_scalar_reduceq!(q, op)
    push!(q.args, Expr(:(=), name(op), reducq))
  end
  q
end
function setup_call_final(ls::LoopSet, q::Expr)
  pushpreamble!(ls, q)
  push!(ls.preamble.args, nothing)
  return ls.preamble
end
function setup_call_debug(ls::LoopSet)
  generate_call(
    ls,
    (false, zero(Int8), zero(Int8), zero(Int8)),
    zero(UInt),
    1,
    true,
    true
  )
end
function setup_call(
  ls::LoopSet,
  q::Expr,
  source::LineNumberNode,
  inline::Bool,
  check_empty::Bool,
  u₁::Int8,
  u₂::Int8,
  v::Int8,
  thread::Int,
  warncheckarg::Int,
  safe::Bool
)
  # We outline/inline at the macro level by creating/not creating an anonymous function.
  # The old API instead was based on inlining or not inline the generated function, but
  # the generated function must be inlined into the initial loop preamble for performance reasons.
  # Creating an anonymous function and calling it also achieves the outlining, while still
  # inlining the generated function into the loop preamble.
  lnns = extract_all_lnns(q)
  pushfirst!(lnns, source)
  call = generate_call(ls, (inline, u₁, u₂, v), thread % UInt, 1, true, false)
  call = check_empty ? check_if_empty(ls, call) : call
  argfailure = make_crashy(make_fast(q))
  if warncheckarg ≠ 0
    warnstring = "$(first(lnns)):\n`LoopVectorization.check_args` on your inputs failed; running fallback `@inbounds @fastmath` loop instead.\nUse `warn_check_args=false`, e.g. `@turbo warn_check_args=false ...`, to disable this warning."
    warning = :(@warn $warnstring)
    warncheckarg > 0 && push!(warning.args, :(maxlog = $warncheckarg))
    argfailure = Expr(:block, warning, argfailure)
  end
  call_check = if safe
    Expr(:&&, check_args_call(ls), check_turbo_safe(ls))
  else
    check_args_call(ls)
  end
  pushprepreamble!(ls, Expr(:if, call_check, call, argfailure))
  prepend_lnns!(ls.prepreamble, lnns)
  return ls.prepreamble
end
