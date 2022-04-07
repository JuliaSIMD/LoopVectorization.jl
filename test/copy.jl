using LoopVectorization, OffsetArrays, Test


@testset "copy" begin

  function copyavx1!(x, y)
    @turbo for i ∈ eachindex(x)
      x[i] = y[i]
    end
  end
  function copy_avx1!(x, y)
    @_avx for i ∈ eachindex(x)
      x[i] = y[i]
    end
  end
  function copyavx2!(x, y)
    @turbo for i ∈ eachindex(x)
      yᵢ = y[i]
      x[i] = yᵢ
    end
  end
  function copy_avx2!(x, y)
    @_avx for i ∈ eachindex(x)
      yᵢ = y[i]
      x[i] = yᵢ
    end
  end
  function offset_copy!(A, B)
    @inbounds for i = 1:size(A, 1), j = 1:size(B, 2)
      A[i, j+2] = B[i, j]
    end
  end
  function offset_copyavx1!(A, B)
    @turbo for i = 1:size(A, 1), j = 1:size(B, 2)
      A[i, j+2] = B[i, j]
    end
  end
  function offset_copy_avx1!(A, B)
    @_avx for i = 1:size(A, 1), j = 1:size(B, 2)
      @inbounds A[i, j+2] = B[i, j]
    end
  end
  function offset_copyavx2!(A, B)
    @turbo for i = 1:size(A, 1), j = 1:size(B, 2)
      Bᵢⱼ = B[i, j]
      A[i, j+2] = Bᵢⱼ
    end
  end
  function offset_copy_avx2!(A, B)
    @_avx for i = 1:size(A, 1), j = 1:size(B, 2)
      Bᵢⱼ = B[i, j]
      A[i, j+2] = Bᵢⱼ
    end
  end
  function make2point3avx!(x)
    a = 1.742416161578685
    b = 1.5
    @turbo for i ∈ eachindex(x)
      x[i] = a^b
    end
  end
  function make2point3_avx!(x)
    @_avx for i ∈ eachindex(x)
      x[i] = 2.3
    end
  end
  function make23avx!(x)
    @turbo for i ∈ eachindex(x)
      @inbounds x[i] = 23
    end
  end
  function make23_avx!(x)
    @_avx for i ∈ eachindex(x)
      x[i] = 23
    end
  end
  function myfillavx!(x, a)
    @turbo for i ∈ eachindex(x)
      x[i] = a
    end
  end
  function myfill_avx!(x, a)
    @_avx for i ∈ eachindex(x)
      x[i] = a
    end
  end
  function issue_256!(A)
    @turbo for i = 1:size(A, 1)
      A[i, (x = 1,).x] = 0
    end
    A
  end

  function reversecopy1!(B, A)
    for i in eachindex(A)
      B[i] = A[11-i]
    end
    B
  end
  function reversecopy1avx!(B, A)
    @turbo for i in eachindex(A)
      B[i] = A[11-i]
    end
    B
  end
  function reversecopy2!(B, A)
    for i in eachindex(B)
      B[i] = A[-i]
    end
    B
  end
  function reversecopy2avx!(B, A)
    @turbo for i in eachindex(B)
      B[i] = A[-i]
    end
    B
  end
  function reversecopy3avx!(B, A)
    n = length(A)
    @turbo for i = 1:n
      B[i] = A[n+1-i]
    end
    B
  end
  function copy3!(B, A)
    @assert (length(B) ≥ 3) && (length(A) ≥ 3)
    @turbo for i = 1:3
      B[i] = A[i]
    end
    B
  end
  function copyselfdot!(s, x)
    m = zero(eltype(x))
    @turbo for i ∈ 1:2
      sᵢ = x[i]
      s[i] = sᵢ
      m += sᵢ * sᵢ
    end
    m
  end
  function scattercopyavx!(H, a, j)
    @turbo for i ∈ eachindex(j), k ∈ eachindex(a)
      H[j[i], k] = a[k]
    end
    H
  end
  function scattercopy!(H, a, j)
    @inbounds for i ∈ eachindex(j), k ∈ eachindex(a)
      H[j[i], k] = a[k]
    end
    H
  end

  function issue279!(output, input, idx)
    @turbo for j in axes(output, 2), i in axes(output, 1)
      output[i, j, idx] = input[1, 1, i, j, idx]
    end
    output
  end

  function issue279_plain!(output, input, idx)
    for j in axes(output, 2), i in axes(output, 1)
      output[i, j, idx] = input[1, 1, i, j, idx]
    end
    output
  end
  function issue280!(dest, src)
    @turbo for i in indices((dest, src), (2, 2))
      dest[1, i] = src[2, i]
      dest[2, i] = src[1, i]
    end
  end

  function fill_with_3rd!(dest, src)
    p = 1
    @turbo for i ∈ eachindex(dest)
      dest[i] = src[3*p]
    end
  end
  function fill_with_r1c3!(dest, src::AbstractMatrix)
    p = 1
    @turbo for i ∈ eachindex(dest)
      dest[i] = src[p, 3*p]
    end
  end
  function collect_turbo(N, ::Type{T}) where {T}
    l = Vector{T}(undef, N)
    @turbo for i ∈ 1:length(l)
      l[i] = i
    end
    return l
  end
  for T ∈ (Float32, Float64, Int32, Int64)
    @show T, @__LINE__
    R = T <: Integer ? (-T(100):T(100)) : T
    x = rand(R, 237)
    q1 = similar(x)
    q2 = similar(x)

    fill!(q2, -999999)
    copyavx1!(q2, x)
    @test x == q2
    fill!(q2, -999999)
    copy_avx1!(q2, x)
    @test x == q2
    fill!(q2, -999999)
    copyavx2!(q2, x)
    @test x == q2
    fill!(q2, -999999)
    copy_avx2!(q2, x)
    @test x == q2
    fill!(q2, -999999)
    @turbo q2 .= x
    @test x == q2
    @test all(iszero, issue_256!(reshape(x, (length(x), 1))))

    B = rand(R, 79, 83)
    A1 = zeros(T, 79, 85)
    A2 = zeros(T, 79, 85)
    offset_copy!(A1, B)
    fill!(A2, 0)
    offset_copyavx1!(A2, B)
    @test A1 == A2
    fill!(A2, 0)
    offset_copyavx2!(A2, B)
    @test A1 == A2
    fill!(A2, 0)
    offset_copy_avx1!(A2, B)
    @test A1 == A2
    fill!(A2, 0)
    offset_copy_avx2!(A2, B)
    @test A1 == A2
    fill_with_3rd!(A1, B)
    @test all(==(B[3]), A1)
    fill_with_3rd!(A1, eachindex(B))
    @test all(==(3), A1)

    fill_with_r1c3!(A1, B)
    @test all(==(B[1, 3]), A1)
    fill_with_r1c3!(A1, LinearIndices(B))# check args fails
    @test all(==(LinearIndices(B)[1, 3]), A1)

    a = rand(R)
    myfillavx!(x, a)
    fill!(q2, a)
    @test x == q2
    a = rand(R)
    myfill_avx!(x, a)
    fill!(q2, a)
    @test x == q2
    a = rand(R)
    myfill_avx!(x, a)
    fill!(q2, a)
    @test x == q2
    a = rand(R)
    myfillavx!(x, a)
    fill!(q2, a)
    @test x == q2
    q2 .= 23
    fill!(q1, -99999)
    make23_avx!(q1)
    @test q2 == q1
    fill!(q1, -99999)
    make23avx!(q1)
    @test q2 == q1
    if T <: Union{Float32,Float64}
      make2point3avx!(x)
      fill!(q2, 2.3)
      @test x == q2
      fill!(x, -999999)
      make2point3_avx!(x)
      @test x == q2
    end
    a = rand(R)
    @turbo x .= a
    fill!(q2, a)
    @test x == q2
    a = rand(R)
    @turbo x .= a
    fill!(q2, a)
    @test x == q2

    @test reversecopy1!(zeros(T, 10), collect(1:10)) ==
          reversecopy1avx!(zeros(T, 10), collect(1:10))
    @test reversecopy2!(zeros(T, 10), OffsetArray(collect(1:10), -10:-1)) ==
          reversecopy2avx!(zeros(T, 10), OffsetArray(collect(1:10), -10:-1))
    @test 177:-1:1 == reversecopy3avx!(zeros(T, 177), collect(1:177))

    x = rand(R, 3)
    y = similar(x)
    @test copy3!(y, x) == x
    fill!(y, 0)
    @test copyselfdot!(y, x) ≈ x[1]^2 + x[2]^2
    @test view(x, 1:2) == view(y, 1:2)

    H0 = zeros(T, 10, 10)
    H1 = zeros(T, 10, 10)
    j = [1, 2, 5, 8]
    a = rand(R, 10)
    @test scattercopyavx!(H0, a, j) == scattercopy!(H1, a, j)

    input = rand(R, 2, 2, 5, 5, 1)
    output = Array{T}(undef, size(input)[3:end]...)
    output_plain = similar(output)

    @test issue279!(output, input, 1) ≈ issue279_plain!(output_plain, input, 1)

    src = rand(R, 2, 17)
    dest = similar(src)
    issue280!(dest, src)
    @test dest ≈ vcat(view(src, 2, :)', view(src, 1, :)')
    if VERSION ≥ v"1.6"
      src2 = reinterpret(reshape, T, Vector{Tuple{T,T}}(undef, 17))
      src2 .= src
      dest2 = reinterpret(reshape, T, Vector{Tuple{T,T}}(undef, 17))
      issue280!(dest2, src2)
      @test dest2 ≈ vcat(view(src, 2, :)', view(src, 1, :)')
    end

    rng = 10:20
    arr1 = zeros(30)
    arr2 = zeros(30)
    ifirst = 5
    @turbo for i in rng
      arr1[i-ifirst] += 1
    end
    @. arr2[rng-ifirst] += 1
    @test arr1 == arr2

    @test collect_turbo(77) == T.(1:77)
  end
end
