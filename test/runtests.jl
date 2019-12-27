using Test
using LoopVectorization


@testset "LoopVectorization.jl" begin

    
    @generated function logsumexp!(r::AbstractArray{T}, x::AbstractArray{T}) where {T}
        quote
            n = length(x)
            length(r) == n || throw(DimensionMismatch())
            isempty(x) && return -T(Inf)
            1 == stride(r,1) == stride(x,1) || throw(error("Arrays not strided"))

            u = maximum(x)                                       # max value used to re-center
            abs(u) == Inf && return any(isnan, x) ? T(NaN) : u   # check for non-finite values

            s = zero(T)
            @vectorize $T for i = 1:n
                tmp = exp(x[i] - u)
                r[i] = tmp
                s += tmp
            end

            invs = inv(s)
            r .*= invs

            return log1p(s-1) + u
        end
    end

    x = collect(1:1_000) ./ 10;
    r = similar(x);

    @test logsumexp!(r, x) ≈ 102.35216846104409

end

exit()
@time using LoopVectorization
using Test
gemmq = :(for i ∈ 1:size(A,1), j ∈ 1:size(B,2)
      Cᵢⱼ = 0.0
      for k ∈ 1:size(A,2)
          Cᵢⱼ += A[i,k] * B[k,j]
      end
      C[i,j] = Cᵢⱼ
      end)

lsgemm = LoopVectorization.LoopSet(gemmq);
U, T = if LoopVectorization.VectorizationBase.REGISTER_COUNT == 16
    (3,4)
else
    (5,5)
end
@test LoopVectorization.choose_order(lsgemm) == (Symbol[:j,:i,:k], U, T)
LoopVectorization.lower(lsgemm)


function mygemm!(C, A, B)
    @inbounds for i ∈ 1:size(A,1), j ∈ 1:size(B,2)
        Cᵢⱼ = 0.0
        @simd ivdep for k ∈ 1:size(A,2)
            Cᵢⱼ += A[i,k] * B[k,j]
        end
        C[i,j] = Cᵢⱼ
    end
end
function mygemmavx!(C, A, B)
    @avx for i ∈ 1:size(A,1), j ∈ 1:size(B,2)
        Cᵢⱼ = 0.0
        for k ∈ 1:size(A,2)
            Cᵢⱼ += A[i,k] * B[k,j]
        end
        C[i,j] = Cᵢⱼ
    end
end

# M, K, N = rand(70:81, 3);
M, K, N = 72, 75, 71;
C = Matrix{Float64}(undef, M, N); A = randn(M, K); B = randn(K, N);
C2 = similar(C);
mygemmavx!(C, A, B)
mygemm!(C2, A, B)
@test all(C .≈ C2)

using BenchmarkTools
@benchmark mygemmavx!($C, $A, $B)
@benchmark mygemm!($C, $A, $B)
using LinearAlgebra
BLAS.set_num_threads(1); BLAS.vendor()
@benchmark mul!($C2, $A, $B)

LoopVectorization.choose_order(lsgemm)
lsgemm.operations

LoopVectorization.choose_tile(lsgemm)
LoopVectorization.choose_unroll_order(lsgemm)

ops = LoopVectorization.oporder(lsgemm);
findall(length.(ops) .!= 0)

dotq = :(for i ∈ eachindex(a,b)
         s += a[i]*b[i]
         end)
lsdot = LoopVectorization.LoopSet(dotq);
@test LoopVectorization.choose_order(lsdot) == (Symbol[:i], 4, -1)
LoopVectorization.lower(lsdot)
lsdot.operations

function mydot(a, b)
    s = 0.0
    @inbounds @simd for i ∈ eachindex(a,b)
        s += a[i]*b[i]
    end
    s
end
function mydotavx(a, b)
    s = 0.0
    @avx for i ∈ eachindex(a,b)
        s += a[i]*b[i]
    end
    s
end
a = rand(400); b = rand(400);
@test mydotavx(a,b) ≈ mydot(a,b)
mydotavx(a,b), mydot(a,b), a' * b
@benchmark mydotavx($a,$b)
@benchmark mydot($a,$b)
@benchmark dot($a,$b)

a = rand(43); b = rand(43);
@benchmark mydotavx($a,$b)
@benchmark mydot($a,$b)
@benchmark dot($a,$b)

selfdotq = :(for i ∈ eachindex(a)
         s += a[i]*a[i]
         end)
lsselfdot = LoopVectorization.LoopSet(selfdotq);
@test LoopVectorization.choose_order(lsselfdot) == (Symbol[:i], 8, -1)
LoopVectorization.lower(lsselfdot)

function myselfdot(a)
    s = 0.0
    @inbounds @simd for i ∈ eachindex(a)
        s += a[i]*a[i]
    end
    s
end
function myselfdotavx(a)
    s = 0.0
    @avx for i ∈ eachindex(a)
        s += a[i]*a[i]
    end
    s
end

a = rand(400);
@test myselfdotavx(a) ≈ myselfdot(a)

@benchmark myselfdotavx($a)
@benchmark myselfdot($a)

@benchmark myselfdotavx($b)
@benchmark myselfdot($b)


vexpq = :(for i ∈ eachindex(a)
          b[i] = exp(a[i])
          end)
lsvexp = LoopVectorization.LoopSet(vexpq);
@test LoopVectorization.choose_order(lsvexp) == (Symbol[:i], 1, -1)
LoopVectorization.lower(lsvexp)

function myvexp!(b, a)
    @inbounds for i ∈ eachindex(a)
        b[i] = exp(a[i])
    end
end
function myvexpavx!(b, a)
    @avx for i ∈ eachindex(a)
        b[i] = exp(a[i])
    end
end
a = randn(127);
b1 = similar(a);
b2 = similar(a);

myvexp!(b1, a)
myvexpavx!(b2, a)
b1'
b2'
all(b1 .≈ b2)
@test all(b1 .≈ b2)

@benchmark myvexp!($b1, $a)
@benchmark myvexpavx!($b2, $a)


vexpsq = :(for i ∈ eachindex(a)
          s += exp(a[i])
          end)
lsvexps = LoopVectorization.LoopSet(vexpsq);
@test LoopVectorization.choose_order(lsvexps) == (Symbol[:i], 1, -1)
LoopVectorization.lower(lsvexps)
lsvexps.operations

function myvexp(a)
    s = 0.0
    @inbounds for i ∈ eachindex(a)
        s += exp(a[i])
    end
    s
end
function myvexpavx(a)
    s = 0.0
    @avx for i ∈ eachindex(a)
        s += exp(a[i])
    end
    s
end

@test myvexp(a) ≈ myvexpavx(a)

@benchmark myvexp($a)
@benchmark myvexpavx($a)

gemvq = :(for i ∈ eachindex(y)
          yᵢ = 0.0
          for j ∈ eachindex(x)
          yᵢ += A[i,j] * x[j]
          end
          y[i] = yᵢ
          end)
lsgemv = LoopVectorization.LoopSet(gemvq);
@test LoopVectorization.choose_order(lsgemv) == (Symbol[:i, :j], 8, -1)
LoopVectorization.lower(lsgemv)


function mygemv!(y, A, x)
    @inbounds for i ∈ eachindex(y)
        yᵢ = 0.0
        @simd for j ∈ eachindex(x)
            yᵢ += A[i,j] * x[j]
        end
        y[i] = yᵢ
    end
end
function mygemvavx!(y, A, x)
    @avx for i ∈ eachindex(y)
        yᵢ = 0.0
        for j ∈ eachindex(x)
            yᵢ += A[i,j] * x[j]
        end
        y[i] = yᵢ
    end
end
A = randn(51, 49);
x = randn(49);
y1 = Vector{Float64}(undef, 51); y2 = similar(y1);
mygemv!(y1, A, x)
mygemvavx!(y2, A, x)

@test all(y1 .≈ y2)

@benchmark mygemv!($y1, $A, $x)
@benchmark mygemvavx!($y2, $A, $x)

subcolq = :(for i ∈ 1:size(A,2), j ∈ eachindex(x)
            B[j,i] = A[j,i] - x[j]
            end)
lssubcol = LoopVectorization.LoopSet(subcolq);
@test LoopVectorization.choose_order(lssubcol) == (Symbol[:j,:i], 4, -1)
LoopVectorization.lower(lssubcol)


## @avx is SLOWER!!!!
## need to fix!
function mysubcol!(B, A, x)
    @inbounds for i ∈ 1:size(A,2)
        @simd for j ∈ eachindex(x)
            B[j,i] = A[j,i] - x[j]
        end
    end
end
function mysubcolavx!(B, A, x)
    @avx for i ∈ 1:size(A,2), j ∈ eachindex(x)
        B[j,i] = A[j,i] - x[j]
    end
end
A = randn(199, 498); x = randn(size(A,1));
B1 = similar(A); B2 = similar(A);

mysubcol!(B1, A, x)
mysubcolavx!(B2, A, x)

@test all(B1 .≈ B2)

@benchmark mysubcol!($B1, $A, $x)
@benchmark mysubcolavx!($B2, $A, $x)

@code_native debuginfo=:none mysubcol!(B1, A, x)
@code_native debuginfo=:none mysubcolavx!(B2, A, x)



# invalid
colsumq = :(for i ∈ 1:size(A,2), j ∈ eachindex(x)
            x[j] += A[j,i]
            end)
colsumq = :(for i ∈ 1:size(A,2), j ∈ eachindex(x)
            x[j] = x[j] + A[j,i]
            end)
# invalid
# Should model aliasing better
# after x[j] is assigned, that must be defined as alias of xj

colsumq = :(for i ∈ 1:size(A,2), j ∈ eachindex(x)
            xj = x[j]
            x[j] = xj + A[j,i]
            end)
# valid
colsumq = :(for i ∈ 1:size(A,2), j ∈ eachindex(x)
            xj = x[j]
            xj = xj + A[j,i]
            x[j] = xj
            end)
#TODO: make this code valid!!!
lscolsum = LoopVectorization.LoopSet(colsumq);
lscolsum
lscolsum.operations

LoopVectorization.choose_order(lscolsum)
@test LoopVectorization.choose_order(lscolsum) == (Symbol[:j,:i], 4, -1)

function mycolsum!(x, A)
    @. x = 0
    @inbounds for i ∈ 1:size(A,2)
        @simd for j ∈ eachindex(x)
            x[j] += A[j,i]
        end
    end
end

mycolsum2q = :(for j ∈ eachindex(x)
        xⱼ = 0.0
        for i ∈ 1:size(A,2)
            xⱼ += A[j,i]
        end
        x[j] = xⱼ
    end)

function mycolsumavx!(x, A)
    @avx for j ∈ eachindex(x)
        xⱼ = 0.0
        for i ∈ 1:size(A,2)
            xⱼ += A[j,i]
        end
        x[j] = xⱼ
    end
end
x1 = similar(x); x2 = similar(x);
mycolsum!(x1, A)
mycolsumavx!(x2, A)

@test all(x1 .≈ x2)
@benchmark mycolsum!($x1, $A)
@benchmark mycolsumavx!($x2, $A)


varq = :(for j ∈ eachindex(s²), i ∈ 1:size(A,2)
         δ = A[j,i] - x̄[j]
         s²[j] += δ*δ
         end)
lsvar = LoopVectorization.LoopSet(varq);
LoopVectorization.choose_order(lsvar)
@test LoopVectorization.choose_order(lsvar) == (Symbol[:j,:i], 5, -1)

function myvar!(s², A, x̄)
    @. s² = 0
    @inbounds for i ∈ 1:size(A,2)
        @simd for j ∈ eachindex(s²)
            δ = A[j,i] - x̄[j]
            s²[j] += δ*δ
        end
    end
end
function myvaravx!(s², A, x̄)
    @avx for j ∈ eachindex(s²)
        s²ⱼ = 0.0
        x̄ⱼ = x̄[j]
        for i ∈ 1:size(A,2)
            δ = A[j,i] - x̄ⱼ
            s²ⱼ += δ*δ
        end
        s²[j] = s²ⱼ
    end
end

x̄ = x1 ./ size(A,2);
myvar!(x1, A, x̄)
myvaravx!(x2, A, x̄)
@test all(x1 .≈ x2)

@benchmark myvar!($x1, $A, $x̄)
@benchmark myvaravx!($x2, $A, $x̄)


using SIMDPirates
function mycolsum2!(
    means::AbstractVector{T}, sample::AbstractArray{T}
) where {T}
    V = VectorizationBase.pick_vector(T)
    W, Wshift = VectorizationBase.pick_vector_width_shift(T)
    WT = VectorizationBase.REGISTER_SIZE
    D, N = size(sample); sample_stride = stride(sample, 2) * sizeof(T)
    @boundscheck if length(means) < D
        throw(BoundsError("Size of sample: ($D,$N); length of preallocated mean vector: $(length(means))"))
    end
    ptr_mean = pointer(means); ptr_smpl = pointer(sample)
    # vNinv = vbroadcast(V, 1/N); vNm1inv = vbroadcast(V, 1/(N-1))
    for _ in 1:(D >>> (Wshift + 2)) # blocks of 4 vectors
        Base.Cartesian.@nexprs 4 i -> Σδ_i = vbroadcast(V, zero(T))
        for n ∈ 0:N-1
            Base.Cartesian.@nexprs 4 i -> δ_i = vload(V, ptr_smpl + WT * (i-1) + n*sample_stride)
            Base.Cartesian.@nexprs 4 i -> Σδ_i = vadd(δ_i, Σδ_i)
        end
        # Base.Cartesian.@nexprs 4 i -> Σδ_i = vmul(vNinv, Σδ_i)
        Base.Cartesian.@nexprs 4 i -> (vstore!(ptr_mean, Σδ_i); ptr_mean += WT)
        ptr_smpl += 4WT
    end
    for _ in 1:((D & ((W << 2)-1)) >>> Wshift) # single vectors
        Σδ_i = vbroadcast(V, zero(T))
        for n ∈ 0:N-1
            δ_i = vload(V, ptr_smpl + n*sample_stride)
            Σδ_i = vadd(δ_i, Σδ_i)
        end
        # Σδ_i = vmul(vNinv, Σδ_i)
        vstore!(ptr_mean, Σδ_i); ptr_mean += WT
        ptr_smpl += WT
    end
    r = D & (W-1)
    if r > 0 # remainder
        mask = VectorizationBase.mask(T, r)
        Σδ_i = vbroadcast(V, zero(T))
        for n ∈ 0:N-1
            δ_i = vload(V, ptr_smpl + n*sample_stride, mask)
            Σδ_i = vadd(δ_i, Σδ_i)
        end
        # Σδ_i = vmul(vNinv, Σδ_i)
        vstore!(ptr_mean, Σδ_i, mask)
    end
    nothing
end



lsgemv.preamble
LoopVectorization.lower(lsgemv)
LoopVectorization.lower_unrolled(lsgemv, 4);

lsgemv.operations

@code_warntype LoopVectorization.choose_order(lsgemm)

lsgemm.operations

lssi = last(ls.operations)
lssi.dependencies
lssi.reduced_deps

lssi.parents

LoopVectorization.num_loops(ls)
LoopVectorization.choose_tile(ls)

LoopVectorization.unitstride(ls.operations[2], :i)

order1 = [:i,:j,:k];
order2 = [:j,:i,:k];
LoopVectorization.evaluate_cost_tile(ls, order1)
LoopVectorization.evaluate_cost_tile(ls, order2)

@code_warntype LoopVectorization.vector_cost(:getindex, 3, 8)
@code_warntype LoopVectorization.cost(first(ls.operations), :i, 3, 8)


LoopVectorization.determine_unroll_factor(ls, [:i,:j])


LoopVectorization.choose_unroll_order(lsgemv)
LoopVectorization.evaluate_cost_unroll(lsgemv, [:i,:j])
LoopVectorization.evaluate_cost_unroll(lsgemv, [:j,:i])

lsgemv.operations

lsvexp.operations

lo = LoopVectorization.LoopOrders(ls);
new_order, state = iterate(lo)
LoopVectorization.evaluate_cost_tile(ls, new_order)

iter = iterate(lo, state)
new_order, state = iter
LoopVectorization.evaluate_cost_tile(ls, new_order)

X = [1.8000554666666666e8, 1.073741824e9, 1.7895697066666666e8, 0.0];
R = [1, 1, 1, 0];
X = [1.79306496e8, 1.7895697066666666e8, 1.7895697066666666e8, 0.0];
R = [1, 1, 1, 0];
LoopVectorization.solve_tilesize(X, R)

using BenchmarkTools
@benchmark LoopVectorization.choose_order($ls)



