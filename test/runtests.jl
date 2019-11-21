using LoopVectorization
using Test

using CpuId, VectorizationBase, SIMDPirates, SLEEFPirates, VectorizedRNG

@generated function estimate_cost_onearg_serial(f::F, N::Int = 512, K = 1_000, ::Type{T} = Float64, ::Val{U} = Val(4)) where {F,T,U}
    quote    
        Base.Cartesian.@nexprs $U u -> s_u = zero(T)
        # s = vbroadcast(V, zero(T))
        x = rand(T, N)
        ptrx = pointer(x)
        ts_start, id_start = cpucycle_id()
        @inbounds for k ∈ 1:K
            i = 1
            for n ∈ 1:N>>$(VectorizationBase.intlog2(U))
                Base.Cartesian.@nexprs $U u -> begin
                    v_u = x[i]
                    i += 1
                    s_u += f(v_u)
                end
            end
        end
        ts_end, id_end = cpucycle_id()
        @assert id_start == id_end
        Base.Cartesian.@nexprs $(U-1) u -> s_1 += s_{u+1}
        (ts_end - ts_start) / (N*K), s_1
    end
end
@generated function estimate_cost_onearg_tworet_serial(f::F, N::Int = 512, K = 1_000, ::Type{T} = Float64, ::Val{U} = Val(4)) where {F,T,U}
    quote    
        Base.Cartesian.@nexprs $U u -> s_u = zero(T)
        # s = vbroadcast(V, zero(T))
        x = rand(T, N)
        ptrx = pointer(x)
        ts_start, id_start = cpucycle_id()
        @inbounds for k ∈ 1:K
            i = 1
            for n ∈ 1:N>>$(VectorizationBase.intlog2(U))
                Base.Cartesian.@nexprs $U u -> begin
                    v_u = x[i]
                    i += 1
                    a_u, b_u = f(v_u)
                    s_u = muladd(a_u,b_u,s_u)
                end
            end
        end
        ts_end, id_end = cpucycle_id()
        @assert id_start == id_end
        Base.Cartesian.@nexprs $(U-1) u -> s_1 += s_{u+1}
        (ts_end - ts_start) / (N*K), s_1
    end
end
@generated function estimate_cost_onearg(f::F, N::Int = 512, K = 1_000, ::Type{T} = Float64, ::Val{U} = Val(4)) where {F,T,U}
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
@generated function estimate_cost_onearg_tworet(f::F, N::Int = 512, K = 1_000, ::Type{T} = Float64, ::Val{U} = Val(4)) where {F,T,U}
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
                    a_u, b_u = f(v_u)
                    s_u = vmuladd(a_u, b_u, s_u)
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
@generated function estimate_cost_twoarg(f::F, N::Int = 512, K = 1_000, ::Type{T} = Float64, ::Val{U} = Val(4)) where {F,T,U}
    W, Wshift = VectorizationBase.pick_vector_width_shift(T)
    if U == 1
        return quote    
            Base.Cartesian.@nexprs $U u -> s_u = vbroadcast(Vec{$W,$T}, one(T))
            # s = vbroadcast(V, zero(T))
            x = rand(T, N << $Wshift)
            ptrx = pointer(x)
            ts_start, id_start = cpucycle_id()
            for k ∈ 1:K
                _ptrx = ptrx
                for n ∈ 1:N>>$(VectorizationBase.intlog2(U))
                    Base.Cartesian.@nexprs $U u -> begin
                        v_u = vload(Vec{$W,$T}, _ptrx)
                        s_u = f(s_u, v_u)
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
    Uh = U >>> 1
    quote    
        Base.Cartesian.@nexprs $(U << 1) u -> s_u = randn(VectorizedRNG.GLOBAL_vPCG, Vec{$W,$T})  #vbroadcast(Vec{$W,$T}, one(T))
        # s = vbroadcast(V, zero(T))
        x = rand(T, N << $Wshift)
        ptrx = pointer(x)
        ts_start, id_start = cpucycle_id()
        for k ∈ 1:K
            _ptrx = ptrx
            for n ∈ 1:N>>$(VectorizationBase.intlog2(U))
                Base.Cartesian.@nexprs $Uh u -> begin
                    v_u = vload(Vec{$W,$T}, _ptrx)
                    _ptrx += VectorizationBase.REGISTER_SIZE
                    v_{u+$Uh} = vload(Vec{$W,$T}, _ptrx)
                    _ptrx += VectorizationBase.REGISTER_SIZE
                    # vv_u = vmul(v_u, v_{u+$Uh})
                    s_u = f(s_u, v_u)
                    s_{u+$Uh} = f(s_{u+$Uh}, v_{u+$Uh})
                    s_{u+$U} = f(s_{u+$U}, v_u)
                    s_{u+$(Uh+U)} = f(s_{u+$(Uh+U)}, v_{u+$Uh})                    
                end
                # v = vload(V, _ptrx)
                # s = vadd(s, f(v))
                # _ptrx += VectorizationBase.REGISTER_SIZE
            end
        end
        ts_end, id_end = cpucycle_id()
        @assert id_start == id_end
        Base.Cartesian.@nexprs $((U<<1)-1) u -> s_1 = vadd(s_1, s_{u+1})
        (ts_end - ts_start) / (2N*K), vsum(s_1)
    end
end
@generated function estimate_cost_threearg(f::F, N::Int = 512, K = 1_000, ::Type{T} = Float64, ::Val{U} = Val(4)) where {F,T,U}
    W, Wshift = VectorizationBase.pick_vector_width_shift(T)
    if U == 1
        return quote
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
                        s_u = f(v_u, v_u, s_u)
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
    Uh = U >>> 1
    quote
        Base.Cartesian.@nexprs $(U<<1) u -> s_u = vbroadcast(Vec{$W,$T}, zero(T))
        # s = vbroadcast(V, zero(T))
        x = rand(T, N << $Wshift)
        ptrx = pointer(x)
        ts_start, id_start = cpucycle_id()
        for k ∈ 1:K
            _ptrx = ptrx
            for n ∈ 1:N>>$(VectorizationBase.intlog2(U))
                Base.Cartesian.@nexprs $Uh u -> begin
                    v_u = vload(Vec{$W,$T}, _ptrx)
                    _ptrx += VectorizationBase.REGISTER_SIZE
                    v_{u+$Uh} = vload(Vec{$W,$T}, _ptrx)
                    _ptrx += VectorizationBase.REGISTER_SIZE
                    s_u = f(v_u, v_u, s_u)
                    s_{u+$Uh} = f(v_{u+$Uh}, v_{u+$Uh}, s_{u+$Uh})
                    s_{u+$U} = f(v_u, v_{u+$Uh}, s_{u+$U})
                    s_{u+$(Uh+U)} = f(v_u, v_{u+$Uh}, s_{u+$(Uh+U)})
                end
                # v = vload(V, _ptrx)
                # s = vadd(s, f(v))
                # _ptrx += VectorizationBase.REGISTER_SIZE
            end
        end
        ts_end, id_end = cpucycle_id()
        @assert id_start == id_end
        Base.Cartesian.@nexprs $((U<<1) - 1) u -> s_1 = vadd(s_1, s_{u+1})
        (ts_end - ts_start) / (2N*K), vsum(s_1)
    end
end
estimate_cost_onearg_serial(exp, 512, 1_000, Float64, Val(1)) # 21
estimate_cost_onearg_serial(exp, 512, 1_000, Float64, Val(2)) # 18.4
estimate_cost_onearg_serial(exp, 512, 1_000, Float64, Val(4)) # 17.5

estimate_cost_onearg_serial(log, 512, 1_000, Float64, Val(1)) # 22
estimate_cost_onearg_serial(log, 512, 1_000, Float64, Val(2)) # 19
estimate_cost_onearg_serial(log, 512, 1_000, Float64, Val(4)) # 19

estimate_cost_onearg_serial(Base.FastMath.sqrt_fast, 512, 1_000, Float64, Val(1)) # 5
estimate_cost_onearg_serial(Base.FastMath.sqrt_fast, 512, 1_000, Float64, Val(2)) # 2.5 # SIMD
estimate_cost_onearg_serial(Base.FastMath.sqrt_fast, 512, 1_000, Float64, Val(4)) # 1.25 # SIMD
@code_native debuginfo=:none estimate_cost_onearg_serial(Base.FastMath.sqrt_fast, 512, 1_000, Float64, Val(4)) # 1.25
estimate_cost_onearg_serial(sqrt, 512, 1_000, Float64, Val(1)) # 5
estimate_cost_onearg_serial(sqrt, 512, 1_000, Float64, Val(2)) # 2.5 # SIMD
estimate_cost_onearg_serial(sqrt, 512, 1_000, Float64, Val(4)) # 1.25 # SIMD
@code_native debuginfo=:none estimate_cost_onearg_serial(sqrt, 512, 1_000, Float64, Val(4)) # 1.25

estimate_cost_onearg_serial(sin, 512, 1_000, Float64, Val(1)) # 18
estimate_cost_onearg_serial(sin, 512, 1_000, Float64, Val(2)) # 15
estimate_cost_onearg_serial(sin, 512, 1_000, Float64, Val(4)) # 15

estimate_cost_onearg_serial(cos, 512, 1_000, Float64, Val(1)) # 19
estimate_cost_onearg_serial(cos, 512, 1_000, Float64, Val(2)) # 16
estimate_cost_onearg_serial(cos, 512, 1_000, Float64, Val(4)) # 16

estimate_cost_onearg_tworet_serial(sincos, 512, 1_000, Float64, Val(1)) # 25
estimate_cost_onearg_tworet_serial(sincos, 512, 1_000, Float64, Val(2)) # 23
estimate_cost_onearg_tworet_serial(sincos, 512, 1_000, Float64, Val(4)) # 22


estimate_cost_onearg(SLEEFPirates.exp, 512, 1_000, Float64, Val(1)) # 28 # 21
estimate_cost_onearg(SLEEFPirates.exp, 512, 1_000, Float64, Val(2)) # 28 # 20
estimate_cost_onearg(SLEEFPirates.exp, 512, 1_000, Float64, Val(4)) # 28 # 19.5

estimate_cost_onearg(SLEEFPirates.log, 512, 1_000, Float64, Val(1)) # 51 cycles # 44
estimate_cost_onearg(SLEEFPirates.log, 512, 1_000, Float64, Val(2)) # 51 cycles # 40
estimate_cost_onearg(SLEEFPirates.log, 512, 1_000, Float64, Val(4)) # 51 cycles # 39

estimate_cost_onearg(SIMDPirates.vsqrt, 512, 1_000, Float64, Val(1)) # 23 cycles # 20
estimate_cost_onearg(SIMDPirates.vsqrt, 512, 1_000, Float64, Val(2)) # 23 cycles # 20
estimate_cost_onearg(SIMDPirates.vsqrt, 512, 1_000, Float64, Val(4)) # 23 cycles # 20

estimate_cost_onearg(SIMDPirates.vinv, 512, 1_000, Float64, Val(1)) # 23 cycles # 13.4
estimate_cost_onearg(SIMDPirates.vinv, 512, 1_000, Float64, Val(2)) # 23 cycles # 13.4
estimate_cost_onearg(SIMDPirates.vinv, 512, 1_000, Float64, Val(4)) # 23 cycles # 13.4

estimate_cost_onearg(SLEEFPirates.sin, 512, 1_000, Float64, Val(1)) #  cycles # 68
estimate_cost_onearg(SLEEFPirates.sin, 512, 1_000, Float64, Val(2)) #  cycles # 66
estimate_cost_onearg(SLEEFPirates.sin, 512, 1_000, Float64, Val(4)) #  cycles # 66

estimate_cost_onearg(SLEEFPirates.cos, 512, 1_000, Float64, Val(1)) #  cycles # 65
estimate_cost_onearg(SLEEFPirates.cos, 512, 1_000, Float64, Val(2)) #  cycles # 68
estimate_cost_onearg(SLEEFPirates.cos, 512, 1_000, Float64, Val(4)) #  cycles # 66

estimate_cost_onearg_tworet(SLEEFPirates.sincos, 512, 1_000, Float64, Val(1)) #  cycles # 71
estimate_cost_onearg_tworet(SLEEFPirates.sincos, 512, 1_000, Float64, Val(2)) #  cycles # 71
estimate_cost_onearg_tworet(SLEEFPirates.sincos, 512, 1_000, Float64, Val(4)) #  cycles # 68

const cz = ntuple(Val(4)) do i Core.VecElement(randn()) end
# @code_native debuginfo=:none
estimate_cost_onearg(x -> SIMDPirates.vmul(x,cz), 1<<9, 10^3, Float64, Val(1)) # 4.5 cycles # 3.35
estimate_cost_onearg(x -> SIMDPirates.vmul(x,cz), 1<<9, 10^3, Float64, Val(2)) # 2 cycles # 1.66
estimate_cost_onearg(x -> SIMDPirates.vmul(x,cz), 1<<9, 10^3, Float64, Val(4)) # 1 cycles # 1
estimate_cost_onearg(x -> SIMDPirates.vmul(x,cz), 1<<9, 10^3, Float64, Val(8)) #  cycles # 0.65

estimate_cost_twoarg(SIMDPirates.vmul, 1<<9, 10^3, Float64, Val(1)) #  cycles # 3.3
estimate_cost_twoarg(SIMDPirates.vmul, 1<<9, 10^3, Float64, Val(2)) #  cycles # 0.97
estimate_cost_twoarg(SIMDPirates.vmul, 1<<9, 10^3, Float64, Val(4)) #  cycles # 0.52
estimate_cost_twoarg(SIMDPirates.vmul, 1<<9, 10^3, Float64, Val(8)) #  cycles # 0.51
estimate_cost_twoarg(SIMDPirates.evmul, 1<<9, 10^3, Float64, Val(8)) #  cycles # 0.51
estimate_cost_twoarg(SIMDPirates.vadd, 1<<9, 10^3, Float64, Val(1)) #  cycles # 3.3
estimate_cost_twoarg(SIMDPirates.vadd, 1<<9, 10^3, Float64, Val(2)) #  cycles # 0.97
estimate_cost_twoarg(SIMDPirates.vadd, 1<<9, 10^3, Float64, Val(4)) #  cycles # 0.52
estimate_cost_twoarg(SIMDPirates.vadd, 1<<9, 10^3, Float64, Val(8)) #  cycles # 0.5
estimate_cost_twoarg(SIMDPirates.evadd, 1<<9, 10^3, Float64, Val(8)) #  cycles # 0.5

@code_native debuginfo=:none estimate_cost_twoarg(SIMDPirates.vmul, 1<<9, 10^3, Float64, Val(8)) #  cycles # 0.64
@code_native debuginfo=:none estimate_cost_twoarg(SIMDPirates.evmul, 1<<9, 10^3, Float64, Val(8)) #  cycles # 0.64


estimate_cost_threearg(SIMDPirates.vmuladd, 1<<9, 10^3, Float64, Val(1)) #  cycles # 3.3
estimate_cost_threearg(SIMDPirates.vmuladd, 1<<9, 10^3, Float64, Val(2)) #  cycles # 0.99
estimate_cost_threearg(SIMDPirates.vmuladd, 1<<9, 10^3, Float64, Val(4)) #  cycles # 0.54
estimate_cost_threearg(SIMDPirates.vmuladd, 1<<9, 10^3, Float64, Val(8)) #  cycles # 0.533
@code_native debuginfo=:none estimate_cost_threearg(SIMDPirates.vmuladd, 1<<9, 10^3, Float64, Val(8)) #  cycles # 0.537
@code_native debuginfo=:none estimate_cost_threearg(SIMDPirates.vfmadd, 1<<9, 10^3, Float64, Val(8)) #  cycles # 0.85

@testset "LoopVectorization.jl" begin
    # Write your own tests here.
end
