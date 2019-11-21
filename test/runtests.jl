using LoopVectorization
using Test

using CpuId, VectorizationBase, SIMDPirates, SLEEFPirates
@generated function estimate_cost(f::F, N::Int = 512, K = 1_000, ::Type{T} = Float64, ::Val{U} = Val(4)) where {F,T,U}
    W, Wshift = VectorizationBase.pick_vector_width_shift(T)
    quote    
        Base.Cartesian.@nexprs $U u -> s_u = vbroadcast(Vec{$W,$T}, zero(T))
        # s = vbroadcast(V, zero(T))
        x = rand(T, N << $Wshift)
        ptrx = pointer(x)
        ts_start, id_start = cpucycle_id()
        for k ∈ 1:K
            _ptrx = ptrx
            for n ∈ 1:N>>$(VectorizationBase.intlog2(U))
                Base.Cartesian.@nexprs $U u -> begin
                    v_u = vload(Vec{$W,$T}, _ptrx)
                    s_u = vadd(s_u, f(v_u))
                    _ptrx += VectorizationBase.REGISTER_SIZE
                end
                # v = vload(V, _ptrx)
                # s = vadd(s, f(v))
                # _ptrx += VectorizationBase.REGISTER_SIZE
            end
        end
        ts_end, id_end = cpucycle_id()
        @assert id_start == id_end
        Base.Cartesian.@nexprs $(U-1) u -> s_1 = vadd(s_1, s_{u+1})
        (ts_end - ts_start) / (N*K), vsum(s_1)
    end
end
estimate_cost(SLEEFPirates.exp, 512, 1_000, Float64, Val(4)) # 28

estimate_cost(SLEEFPirates.log, 512, 1_000, Float64, Val(1)) # 51 cycles
estimate_cost(SLEEFPirates.log, 512, 1_000, Float64, Val(2)) # 51 cycles
estimate_cost(SLEEFPirates.log, 512, 1_000, Float64, Val(4)) # 51 cycles
estimate_cost(SIMDPirates.vsqrt, 512, 1_000, Float64, Val(1)) # 23 cycles
estimate_cost(SIMDPirates.vsqrt, 512, 1_000, Float64, Val(2)) # 23 cycles
estimate_cost(SIMDPirates.vsqrt, 512, 1_000, Float64, Val(4)) # 23 cycles
estimate_cost(SIMDPirates.vinv, 512, 1_000, Float64, Val(1)) # 23 cycles
estimate_cost(SIMDPirates.vinv, 512, 1_000, Float64, Val(2)) # 23 cycles
estimate_cost(SIMDPirates.vinv, 512, 1_000, Float64, Val(4)) # 23 cycles

const cz = ntuple(Val(4)) do i Core.VecElement(randn()) end
# @code_native debuginfo=:none
estimate_cost(x -> SIMDPirates.vmul(x,cz), 1<<9, 10^3, Float64, Val(1)) # 4.5 cycles
estimate_cost(x -> SIMDPirates.vmul(x,cz), 1<<9, 10^3, Float64, Val(2)) # 2 cycles
estimate_cost(x -> SIMDPirates.vmul(x,cz), 1<<9, 10^3, Float64, Val(4)) # 1 cycles

@testset "LoopVectorization.jl" begin
    # Write your own tests here.
end
