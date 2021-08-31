@testset "Special Functions" begin
  vexpq = :(for i ∈ eachindex(a)
              b[i] = exp(a[i])
            end)
  lsvexp = LoopVectorization.loopset(vexpq);
  @test LoopVectorization.choose_order(lsvexp) == (Symbol[:i], :i, Symbol("##undefined##"), :i, 2, -1)

  function myvexp!(b, a)
    @inbounds for i ∈ eachindex(a)
      b[i] = exp(a[i])
    end
  end
  function myvexpavx!(b, a)
    @turbo for i ∈ eachindex(a)
      b[i] = Base.exp(a[i])
    end
  end
  function myvexp_avx!(b, a)
    @_avx for i ∈ eachindex(a)
      b[i] = exp(a[i])
    end
  end
  function offset_exp!(A, B)
    @turbo for i=1:size(A,1), j=1:size(B,2)
      A[i,j] = exp(B[i,j])
    end
  end
  function offset_expavx!(A, B)
    @turbo for i=1:size(A,1), j=1:size(B,2)
      A[i,j] = exp(B[i,j])
    end
  end
  function offset_exp_avx!(A, B)
    @_avx for i=1:size(A,1), j=1:size(B,2)
      A[i,j] = exp(B[i,j])
    end
  end

  vexpsq = :(for i ∈ eachindex(a)
               s += exp(a[i])
             end)
  lsvexps = LoopVectorization.loopset(vexpsq);
  @test LoopVectorization.choose_order(lsvexps) == (Symbol[:i], :i, Symbol("##undefined##"), :i, 2, -1)

  function myvexp(a)
    s = zero(eltype(a))
    @inbounds for i ∈ eachindex(a)
      s += exp(a[i])
    end
    s
  end
  function myvexpavx(a)
    s = zero(eltype(a))
    @turbo for i ∈ eachindex(a)
      s += exp(a[i])
    end
    s
  end

  function myvexp_avx(a)
    s = zero(eltype(a))
    @_avx for i ∈ eachindex(a)
      s += exp(a[i])
    end
    s
  end
  function trianglelogdetavx(L)
    ld = zero(eltype(L))
    @turbo for i ∈ 1:size(L,1)
      ld += log(L[i,i])
    end
    ld
  end
  function trianglelogdet_avx(L)
    ld = zero(eltype(L))
    @_avx for i ∈ 1:size(L,1)
      ld += log(L[i,i])
    end
    ld
  end
  function testrepindshigherdim_avx(L)
    ld = zero(eltype(L))
    @_avx for i ∈ axes(L,1), j ∈ axes(L,2)
      ld += log(L[i,j,i])
    end
    ld
  end
  function testrepindshigherdimavx(L)
    ld = zero(eltype(L))
    @turbo for i ∈ axes(L,1), j ∈ axes(L,2)
      ld += log(L[i,j,i])
    end
    ld
  end
  ldq = :(for i ∈ 1:size(L,1)
            ld += log(L[i,i])
          end)
  lsld = LoopVectorization.loopset(ldq);
  @test LoopVectorization.choose_order(lsld) == (Symbol[:i], :i, Symbol("##undefined##"), :i, 2, -1)

  function calc_sins!(res::AbstractArray{T}) where {T}
    code_phase_delta = T(0.01)
    @inbounds for i ∈ eachindex(res)
      res[i] = sin(i * code_phase_delta)
    end
  end
  function calc_sinsavx!(res::AbstractArray{T}) where {T}
    code_phase_delta = T(0.01)
    @turbo for i ∈ eachindex(res)
      res[i] = sin(i * code_phase_delta)
    end
  end
  function calc_sins_avx!(res::AbstractArray{T}) where {T}
    code_phase_delta = T(0.01)
    @_avx for i ∈ eachindex(res)
      res[i] = sin(i * code_phase_delta)
    end
  end
  
  function logsumexp!(r::AbstractArray{T}, x::AbstractArray{T}) where {T}
    n = length(x)
    length(r) == n || throw(DimensionMismatch())
    isempty(x) && return -T(Inf)
    1 == LinearAlgebra.stride1(r) == LinearAlgebra.stride1(x) || throw(error("Arrays not strided"))

    u = maximum(x)                                       # max value used to re-center
    abs(u) == Inf && return any(isnan, x) ? T(NaN) : u   # check for non-finite values
    s = zero(T)
    @inbounds for i = 1:n
      tmp = exp(x[i] - u)
      r[i] = tmp
      s += tmp
    end
    invs = inv(s)
    r .*= invs

    return log1p(s-1) + u
  end
  function logsumexpavx!(r::AbstractArray{T}, x::AbstractArray{T}) where {T}
    n = length(x)
    length(r) == n || throw(DimensionMismatch())
    isempty(x) && return -T(Inf)
    1 == LinearAlgebra.stride1(r) == LinearAlgebra.stride1(x) || throw(error("Arrays not strided"))

    u = maximum(x)                                       # max value used to re-center
    abs(u) == Inf && return any(isnan, x) ? T(NaN) : u   # check for non-finite values
    s = zero(T)
    @turbo for i = 1:n
      tmp = exp(x[i] - u)
      r[i] = tmp
      s += tmp
    end

    invs = inv(s)
    r .*= invs

    return log1p(s-1) + u
  end
  function logsumexp_avx!(r::AbstractArray{T}, x::AbstractArray{T}) where {T}
    n = length(x)
    length(r) == n || throw(DimensionMismatch())
    isempty(x) && return -T(Inf)
    1 == LinearAlgebra.stride1(r) == LinearAlgebra.stride1(x) || throw(error("Arrays not strided"))

    u = maximum(x)                                       # max value used to re-center
    abs(u) == Inf && return any(isnan, x) ? T(NaN) : u   # check for non-finite values
    s = zero(T)
    @_avx for i = 1:n
      tmp = exp(x[i] - u)
      r[i] = tmp
      s += tmp
    end

    invs = inv(s)
    r .*= invs

    return log1p(s-1) + u
  end
  # feq = :(for i = 1:n
  #         tmp = exp(x[i] - u)
  #         r[i] = tmp
  #         s += tmp
  #         end)
  # lsfeq = LoopVectorization.loopset(feq);
  # # lsfeq.operations

  function vpow0!(y, x)
    @turbo for i ∈ eachindex(y, x)
      y[i] = x[i] ^ 0
    end; y
  end
  function vpown1!(y, x)
    @turbo for i ∈ eachindex(y, x)
      y[i] = x[i] ^ -1
    end; y
  end
  function vpow1!(y, x)
    @turbo for i ∈ eachindex(y, x)
      y[i] = x[i] ^ 1
    end; y
  end
  function vpown2!(y, x)
    @turbo for i ∈ eachindex(y, x)
      y[i] = x[i] ^ -2
    end; y
  end
  function vpow2!(y, x)
    @turbo for i ∈ eachindex(y, x)
      y[i] = x[i] ^ 2
    end; y
  end
  function vpown3!(y, x)
    @turbo for i ∈ eachindex(y, x)
      y[i] = x[i] ^ -3
    end; y
  end
  function vpow3!(y, x)
    @turbo for i ∈ eachindex(y, x)
      y[i] = x[i] ^ 3
    end; y
  end
  function vpown4!(y, x)
    @turbo for i ∈ eachindex(y, x)
      y[i] = x[i] ^ -4
    end; y
  end
  function vpow4!(y, x)
    @turbo for i ∈ eachindex(y, x)
      y[i] = x[i] ^ 4
    end; y
  end
  function vpown5!(y, x)
    @turbo for i ∈ eachindex(y, x)
      y[i] = x[i] ^ -5
    end; y
  end
  q = :(for i ∈ eachindex(y, x)
          y[i] = x[i] ^ -5
        end);
  ls = LoopVectorization.loopset(q);
  q2 = :(for i ∈ eachindex(y, x)
           y[i] = x[i] ^ 5
         end);
  ls2 = LoopVectorization.loopset(q2);
  
  function vpow5!(y, x)
    @turbo for i ∈ eachindex(y, x)
      y[i] = x[i] ^ 5
    end; y
  end
  function vpowf!(y, x)
    @turbo for i ∈ eachindex(y, x)
      y[i] = x[i] ^ 2.3
    end; y
  end
  function vpowf!(y, x, p::Number)
    @turbo for i ∈ eachindex(y, x)
      y[i] = x[i] ^ p
    end; y
  end
  function vpowf!(y, x, p::AbstractArray)
    @turbo for i ∈ eachindex(y, x)
      y[i] = x[i] ^ p[i]
    end; y
  end
  @generated function vpow!(y, x, ::Val{p}) where {p}
    quote
      @turbo for i ∈ eachindex(y,x)
        y[i] = x[i] ^ $p
      end
      return y
    end
  end
  
  function csetanh!(y, z, x)
    for j in axes(x, 2)
      for i = axes(x, 1)
        t2 = inv(tanh(x[i, j]))
        t1 = tanh(x[i, j])
        y[i, j] = z[i, j] * (-(1 - t1 ^ 2) * t2)
      end
    end
    y
  end    
  function csetanhavx!(y, z, x)
    @turbo for j in axes(x, 2)
      for i = axes(x, 1)
        t2 = inv(tanh(x[i, j]))
        t1 = tanh(x[i, j])
        y[i, j] = z[i, j] * (-(1 - t1 ^ 2) * t2)
      end
    end
    y
  end    

  function transposedvectoraccess(x::AbstractVector{T}) where T
    N = length(x)
    ent = zeros(T, N)
    x isa AbstractVector && (x = x')
    for i = 1:N, j = 1:size(x,1)
      ent[i] += x[j,i]*log(x[j,i])
    end
    ent
  end
  function transposedvectoraccessavx(x::AbstractVector{T}) where T
    N = length(x)
    ent = zeros(T, N)
    x isa AbstractVector && (x = x')
    @turbo for i = 1:N, j = 1:size(x,1)
      ent[i] += x[j,i]*log(x[j,i])
    end
    ent
  end

  function vsincos(x)
    y = Matrix{eltype(x)}(undef, length(x), 2)
    for i = eachindex(x)
      y[i,1], y[i,2] = sincos(x[i])
    end
    return y
  end
  function vsincosavx(x)
    y = Matrix{eltype(x)}(undef, length(x), 2)
    @turbo for i = eachindex(x)
      y[i,1], y[i,2] = sincos(x[i])
    end
    return y
  end
  function sincosdot(x)
    a = zero(eltype(x))
    for i ∈ eachindex(x)
      s, c = sincos(x[i])
      a += s * c
    end
    a
  end
  function sincosdotavx(x)
    a = zero(eltype(x))
    @turbo for i ∈ eachindex(x)
      s, c = sincos(x[i])
      a += s * c
    end
    a
  end

  function sin_sum_3loop!(u, x, y, z)
    @turbo for k in 1:length(z)
      for j in 1:length(y)
        for i in 1:length(x)
          u[i, j, k] = sin(x[i]) + sin(y[j]) + sin(z[k])
        end 
      end 
    end 
  end
  function sin_sum_3loop_split!(u, x, y, z)
    sx = similar(x); sy = similar(y); sz = similar(z);
    @turbo for k in 1:length(z)
      for j in 1:length(y)
        for i in 1:length(x)
          sxi = sin(x[i])
          syj = sin(y[j])
          szk = sin(z[k])
          sx[i] = sxi; sy[j] = syj; sz[k] = szk;
          u[i, j, k] = sxi + syj + szk
        end 
      end 
    end 
  end

  for T ∈ (Float32, Float64)
    @show T, @__LINE__
    a = randn(T, 127);
    b1 = similar(a);
    b2 = similar(a);

    myvexp!(b1, a)
    myvexpavx!(b2, a)
    @test b1 ≈ b2
    fill!(b2, -999.9); myvexp_avx!(b2, a)
    @test b1 ≈ b2
    s = myvexp(a)
    @test s ≈ myvexpavx(a)
    @test s ≈ myvexp_avx(a)
    @test b1 ≈ @turbo exp.(a)

    @test vsincos(a) ≈ vsincosavx(a)
    @test sincosdot(a) ≈ sincosdotavx(a)
    
    A = rand(T, 73, 73);
    ld = logdet(UpperTriangular(A))
    @test ld ≈ trianglelogdetavx(A)
    @test ld ≈ trianglelogdet_avx(A)
    Adim3 = rand(T, 37, 13, 37);
    ld = sum(i -> logdet(UpperTriangular(@view(Adim3[:,i,:]))), axes(Adim3,2))
    @test ld ≈ testrepindshigherdimavx(Adim3)
    @test ld ≈ testrepindshigherdim_avx(Adim3)

    x = rand(T, 999);
    r1 = similar(x);
    r2 = similar(x);
    lse = logsumexp!(r1, x);
    @test logsumexpavx!(r2, x) ≈ lse
    @test r1 ≈ r2
    fill!(r2, T(NaN));
    @test logsumexp_avx!(r2, x) ≈ lse
    @test r1 ≈ r2

    calc_sins!(r1)
    calc_sinsavx!(r2)
    @test r1 ≈ r2
    fill!(r2, NaN); calc_sins_avx!(r2)
    @test r1 ≈ r2

    N,M = 47,53
    B = reshape(cumsum(ones(T, 3N)),N,:)
    A1 = zeros(T, N, M)
    A2 = zeros(T, N, M)
    offset_exp!(A1, B)
    offset_expavx!(A2, B)
    @test A1 ≈ A2
    fill!(A2, 0); offset_exp_avx!(A2, B)
    @test A1 ≈ A2

    

    @test all(isone, vpow0!(r1, x))
    @test vpown1!(r1, x) ≈ map!(inv, r2, x)
    @test vpow1!(r1, x) == x
    @test vpown2!(r1, x) ≈ map!(abs2 ∘ inv, r2, x)
    @test vpow2!(r1, x) ≈ map!(abs2, r2, x)
    @test vpown3!(r1, x) ≈ (r2 .= x .^ -3)
    @test vpow3!(r1, x) ≈ (r2 .= x .^ 3)
    @test vpown4!(r1, x) ≈ (r2 .= x .^ -4)
    @test vpow4!(r1, x) ≈ (r2 .= x .^ 4)
    @test vpown5!(r1, x) ≈ (r2 .= x .^ -5)
    @test vpow5!(r1, x) ≈ (r2 .= x .^ 5)
    @test vpowf!(r1, x) ≈ (r2 .= x .^ 2.3)
    @test vpowf!(r1, x, -1.7) ≈ (r2 .= x .^ -1.7)
    p = randn(length(x));
    @test vpowf!(r1, x, x) ≈ (r2 .= x .^ x)
    @test vpow!(r1, x, Val(0.75)) ≈ (r2 .= x .^ 0.75)
    @test vpow!(r1, x, Val(2/3)) ≈ (r2 .= x .^ (2/3))
    vpow!(r1, x, Val(0.5)); r2 .= Base.sqrt_llvm.(x)
    # if Bool(!LoopVectorization.VectorizationBase.has_feature(Val(:x86_64_avx2)))
      for i ∈ eachindex(x)
        # @test abs(r1[i] - r2[i]) ≤ eps(r1[i])
        @test abs(r1[i] - r2[i]) ≤ 2eps(r1[i])
      end
    # else
    #   @test r1 == r2
    # end
    @test vpow!(r1, x, Val(1/4)) ≈ (r2 .= x .^ (1/4))
    @test vpow!(r1, x, Val(4.5)) ≈ (r2 .= x .^ 4.5)

    X = rand(T, N, M); Z = rand(T, N, M);
    Y1 = similar(X); Y2 = similar(Y1);
    @test csetanh!(Y1, X, Z) ≈ csetanhavx!(Y2, X, Z)

    x = rand(T, 97);
    @test transposedvectoraccessavx(x) ≈ transposedvectoraccess(x)

    itot = 47;
    dx = 1. / itot;
    x = dx*collect(0:itot-1); y = dx*collect(0:itot-1); z = dx*collect(0:itot-1);
    u = zeros(itot+8, itot+8, itot+8);
    uv = @view u[5:5+itot-1, 5:5+itot-1, 5:5+itot-1];
    sin_sum_3loop!(uv, x, y, z);
    uv2 = @view similar(u)[5:5+itot-1, 5:5+itot-1, 5:5+itot-1];
    sin_sum_3loop_split!(uv2, x, y, z);
    @test uv ≈ uv2 ≈ (identity(sin.(x)) .+ identity((sin.(y))')) .+ identity(reshape(sin.(z), (1, 1, length(z))))
  end
end
