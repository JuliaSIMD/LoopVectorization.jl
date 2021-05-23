using VectorizationBase, LoopVectorization
using VectorizationBase: data

# @generated to use VectorizationBase's API for supporting 1.5 and 1.6+
@generated function readcyclecounter()
    decl = "declare i64 @llvm.readcyclecounter()"
    instr = "%res = call i64 @llvm.readcyclecounter()\nret i64 %res"
    VectorizationBase.llvmcall_expr(decl, instr, :Int64, :(Tuple{}), "i64", String[], Symbol[])
end

@generated function volatile(x::Vec{W,T}) where {W,T}
    typ = VectorizationBase.LLVM_TYPES[T]
    vtyp = "<$W x $typ>"

    suffix = T == Float32 ? "ps" : "pd"
    sideeffect_str = """%res = call <$W x $(typ)> asm sideeffect "", "=v,v"(<$W x $(typ)> %0)
                               ret <$W x $(typ)> %res"""
    quote
        $(Expr(:meta, :inline))
        Vec(Base.llvmcall($sideeffect_str, NTuple{$W,Core.VecElement{$T}}, Tuple{NTuple{$W,Core.VecElement{$T}}}, VectorizationBase.data(x)))
    end
end
@inline volatile(x::VecUnroll) = VecUnroll(VectorizationBase.fmap(volatile, data(x)))
@inline volatile(x::Tuple) = map(volatile, x)
# @generated function volatile(x::Vec{W,T}, x::Vec{W,T}) where {W,T}
#     typ = VectorizationBase.LLVM_TYPES[T]
#     vtyp = "<$W x $typ>"

#     suffix = T == Float32 ? "ps" : "pd"
#     sideeffect_str = """%res = call <$W x $(typ)> asm sideeffect "", "=v,v"(<$W x $(typ)> %0, <$W x $(typ)> %1)
#                                ret <$W x $(typ)> %res"""
#     quote
#         $(Expr(:meta, :inline))
#         Vec(Base.llvmcall($sideeffect_str, NTuple{$W,Core.VecElement{$T}}, Tuple{NTuple{$W,Core.VecElement{$T}}}, VectorizationBase.data(x)))
#     end
# end

num_vectors(::VecUnroll{N}) where {N} = N+1
num_vectors(::Vec) = 1
function unrolltest(f::F, vs::Vararg{Any,K}) where {F,K}
    cc = readcyclecounter()
    # num_iter = 1_048_576
    num_iter = 4_194_304
    for i ∈ 1:num_iter
        volatile(f(map(volatile, vs)...))
    end
    cycles = readcyclecounter() - cc
    cycles / (num_vectors(first(vs)) * num_iter)
end

# @generated function vapply!(f::F, y, x, ::Val{U}) where {F,U}
#     quote
#         @turbo unroll=$U for j ∈ 1:1024
#             y[j] = f(x[j])
#         end
#     end
# end

# vector_init(::Val{N}, ::Type{T}) where {N,T} = VectorizationBase.zero_vecunroll(StaticInt(N), pick_vector_width(T), T, VectorizationBase.register_size())
# vector_init(::Val{1}, ::Type{T}) where {T} = VectorizationBase.vzero(pick_vector_width(T), T)

# @generated function unrolltest(f::F, x::AbstractVector{T}, ::Val{U}) where {F,U,T}
#     quote
#         cc = readcyclecounter()
#         for i ∈ 1:8192
#             s = vector_init(Val{$U}(), $T)
#             @turbo unroll=$U for j ∈ 1:512
#                 s += f(x[j])
#             end
#             volatile(s)
#         end
#         cycles = readcyclecounter() - cc
#         pick_vector_width(T) * cycles / (512 * 8192)
#     end
# end


# @generated function unrolltest!(f::F, y::AbstractVector{T}, x::AbstractVector{T}, ::Val{U}) where {F,U,T}
#     quote
#         cc = readcyclecounter()
#         for i ∈ 1:8192
#             @turbo unroll=$U for j ∈ 1:512
#                 y[j] = f(x[j])
#             end
#         end
#         cycles = readcyclecounter() - cc
#         pick_vector_width(T) * cycles / (512 * 8192)
#     end
# end

let
    vx = Vec(ntuple(_ -> 10randn(), pick_vector_width(Float64))...);
    vu2 = VectorizationBase.VecUnroll(ntuple(_ -> Vec(ntuple(_ -> 10randn(), pick_vector_width(Float64))...), Val(2)));
    vu4 = VectorizationBase.VecUnroll(ntuple(_ -> Vec(ntuple(_ -> 10randn(), pick_vector_width(Float64))...), Val(4)));
    vu8 = VectorizationBase.VecUnroll(ntuple(_ -> Vec(ntuple(_ -> 10randn(), pick_vector_width(Float64))...), Val(8)));
    for unaryf ∈ [log, log2, log10, log1p, exp, exp2, exp10, expm1, sin, cos]
        rt1 = unrolltest(f, vx)
        rt2 = unrolltest(f, vu2)
        rt4 = unrolltest(f, vu4)
        rt8 = unrolltest(f, vu8)
    end
    for binaryf ∈ [+, *, ^]
        rt1 = unrolltest(f, vx, vx)
        rt2 = unrolltest(f, vu2, vu2)
        rt4 = unrolltest(f, vu4, vu4)
        rt8 = unrolltest(f, vu8, vu8)
    end
end

let
    f, io = mktemp()
    W = Int(VectorizationBase.pick_vector_width(Float64))
    code_native(io, exp, (VecUnroll{1,W,Float64,Vec{W,Float64}},); debuginfo=:none)
    close(io)
    run(`llvm-mca -mcpu=$(Sys.CPU_NAME) -output-asm-variant=1 -bottleneck-analysis $f`)
end


