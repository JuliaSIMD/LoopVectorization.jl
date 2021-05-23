


function dot_simd(a::AbstractVector, b::AbstractVector)
    s = zero(eltype(a))
    @fastmath @inbounds @simd for i ∈ eachindex(a)
        s += a[i]' * b[i]
    end
    s
end
function cdot_mat(ca::AbstractVector{Complex{T}}, cb::AbstractVector{Complex{T}}) where {T}
    a = reinterpret(reshape, T, ca)
    b = reinterpret(reshape, T, cb)
    re = zero(T); im = zero(T)
    @turbo for i ∈ axes(a,2)
        re += a[1,i] * b[1,i] + a[2,i] * b[2,i]
        im += a[1,i] * b[2,i] - a[2,i] * b[1,i]
    end
    Complex(re, im)
end
function cdot_affine(ca::AbstractVector{Complex{T}}, cb::AbstractVector{Complex{T}}) where {T}
    a = reinterpret(T, ca);
    b = reinterpret(T, cb);
    re = zero(T); im = zero(T)
    # with a multiplier, we go from `i = 1 -> 2i = 2` to `i = 0 -> 2i = 0
    # 2(i+1-1) = 2i + 2 - 2, so....
    @turbo for i ∈ 1:length(a)>>>1
        re += a[2i-1] * b[2i-1] + a[2i] * b[2i  ]
        im += a[2i-1] * b[2i  ] - a[2i] * b[2i-1]
    end
    Complex(re, im)
end
function cdot_stride(ca::AbstractVector{Complex{T}}, cb::AbstractVector{Complex{T}}) where {T}
    a = reinterpret(T, ca);
    b = reinterpret(T, cb);
    re = zero(T); im = zero(T)
    @turbo for i ∈ 1:2:length(a)
        re += a[i] * b[i  ] + a[i+1] * b[i+1]
        im += a[i] * b[i+1] - a[i+1] * b[i  ]
    end
    Complex(re, im)
end
function qdot_simd(x::AbstractVector{NTuple{4,T}}, y::AbstractVector{NTuple{4,T}}) where {T}
    a = zero(T)
    b = zero(T)
    c = zero(T)
    d = zero(T)
    @fastmath @inbounds @simd for i ∈ eachindex(x)
        a₁, b₁, c₁, d₁ = x[i]
        a₂, b₂, c₂, d₂ = y[i]
        a += a₁*a₂ + b₁*b₂ + c₁*c₂ + d₁*d₂
        b += a₁*b₂ - b₁*a₂ - c₁*d₂ + d₁*c₂
        c += a₁*c₂ + b₁*d₂ - c₁*a₂ - d₁*b₂
        d += a₁*d₂ - b₁*c₂ + c₁*b₂ - d₁*a₂
    end
    (a,b,c,d)
end
function qdot_mat(x::AbstractMatrix, y::AbstractMatrix)
    a = zero(eltype(x))
    b = zero(eltype(x))
    c = zero(eltype(x))
    d = zero(eltype(x))
    @turbo for i ∈ axes(x,2)
        a₁ = x[1,i]
        b₁ = x[2,i]
        c₁ = x[3,i]
        d₁ = x[4,i]
        a₂ = y[1,i]
        b₂ = y[2,i]
        c₂ = y[3,i]
        d₂ = y[4,i]
        a += a₁*a₂ + b₁*b₂ + c₁*c₂ + d₁*d₂
        b += a₁*b₂ - b₁*a₂ - c₁*d₂ + d₁*c₂
        c += a₁*c₂ + b₁*d₂ - c₁*a₂ - d₁*b₂
        d += a₁*d₂ - b₁*c₂ + c₁*b₂ - d₁*a₂
    end
    (a,b,c,d)
end
function qdot_affine(x::AbstractVector, y::AbstractVector)
    a = zero(eltype(x))
    b = zero(eltype(x))
    c = zero(eltype(x))
    d = zero(eltype(x))
    @turbo for i ∈ 1:length(x)>>2
        a₁ = x[4i-3]
        b₁ = x[4i-2]
        c₁ = x[4i-1]
        d₁ = x[4i]
        a₂ = y[4i-3]
        b₂ = y[4i-2]
        c₂ = y[4i-1]
        d₂ = y[4i]
        a += a₁*a₂ + b₁*b₂ + c₁*c₂ + d₁*d₂
        b += a₁*b₂ - b₁*a₂ - c₁*d₂ + d₁*c₂
        c += a₁*c₂ + b₁*d₂ - c₁*a₂ - d₁*b₂
        d += a₁*d₂ - b₁*c₂ + c₁*b₂ - d₁*a₂
    end
    (a,b,c,d)
end
function qdot_stride(x::AbstractVector, y::AbstractVector)
    a = zero(eltype(x))
    b = zero(eltype(x))
    c = zero(eltype(x))
    d = zero(eltype(x))
    @turbo for i ∈ 1:4:length(x)
        a₁ = x[i]
        b₁ = x[i+1]
        c₁ = x[i+2]
        d₁ = x[i+3]
        a₂ = y[i]
        b₂ = y[i+1]
        c₂ = y[i+2]
        d₂ = y[i+3]
        a += a₁*a₂ + b₁*b₂ + c₁*c₂ + d₁*d₂
        b += a₁*b₂ - b₁*a₂ - c₁*d₂ + d₁*c₂
        c += a₁*c₂ + b₁*d₂ - c₁*a₂ - d₁*b₂
        d += a₁*d₂ - b₁*c₂ + c₁*b₂ - d₁*a₂
    end
    (a,b,c,d)
end
function cmatmul_array!(C::AbstractArray{T,3}, A::AbstractArray{T,3}, B::AbstractArray{T,3}) where {T}
    @turbo for n ∈ indices((C,B),3), m ∈ indices((C,A),2)
        Cre = zero(T)
        Cim = zero(T)
        for k ∈ indices((A,B),(3,2))
            Cre += A[1,m,k] * B[1,k,n] - A[2,m,k] * B[2,k,n]
            Cim += A[1,m,k] * B[2,k,n] + A[2,m,k] * B[1,k,n]
        end
        C[1,m,n] = Cre
        C[2,m,n] = Cim
    end
end

function issue209(M, G, J, H, B, ϕ)
    # tmp = similar(ϕ, G-1, (2*J+1)*(H + 1));
    tmp = view(fill(eltype(ϕ)(123456789), G+15, (2*J+1)*(H + 1)+16), 9:G+7, 9:(2*J+1)*(H + 1)+8);
    Bf = reinterpret(reshape, Float64, B);
    ϕf = reinterpret(reshape, Float64, ϕ);
    tmpf = reinterpret(reshape, Float64, tmp);
    jmax = 2*J + 1
    # B is being indexed via ptr offsetting
    # thus B's initial `gesp`ing must set it up for this
    # currently, it isn't because  `jj` and `hh` are loop induct vars
    for mm = 1:M
        m_idx = M + 2 - mm
        @turbo for hh = 1:H+1
            h_idx = (hh - 1)*jmax
            for jj = 1:jmax, gg = 1:G-1
                tmpf[1, gg, jj + h_idx] = ϕf[1, jj, gg+1, hh, m_idx] +
                                          Bf[1, jj, gg,   hh, m_idx]
                tmpf[2, gg, jj + h_idx] = ϕf[2, jj, gg+1, hh, m_idx] +
                                          Bf[2, jj, gg,   hh, m_idx]
            end
        end
    end
    parent(tmp)
end
function issue209_noavx(M, G, J, H, B, ϕ)
    tmp = view(fill(eltype(ϕ)(123456789), G+15, (2*J+1)*(H + 1)+16), 9:G+7, 9:(2*J+1)*(H + 1)+8);
    Bf = reinterpret(reshape, Float64, B);
    ϕf = reinterpret(reshape, Float64, ϕ);
    tmpf = reinterpret(reshape, Float64, tmp);
    jmax = 2*J + 1
    for mm = 1:M
        m_idx = M + 2 - mm
        for hh = 1:H+1
            h_idx = (hh - 1)*jmax
            for jj = 1:jmax, gg = 1:G-1
                tmpf[1, gg, jj + h_idx] = ϕf[1, jj, gg+1, hh, m_idx] +
                                          Bf[1, jj, gg,   hh, m_idx]
                tmpf[2, gg, jj + h_idx] = ϕf[2, jj, gg+1, hh, m_idx] +
                                          Bf[2, jj, gg,   hh, m_idx]
            end
        end
    end
    parent(tmp)
end

@testset "shuffles load/stores" begin
    @show @__LINE__
    for i ∈ 1:128
        ac = rand(Complex{Float64}, i);
        bc = rand(Complex{Float64}, i);
        dsimd = dot_simd(ac, bc)
        if VERSION ≥ v"1.6.0-rc1"
            @test dsimd ≈ cdot_mat(ac, bc)
        end
        @test dsimd ≈ cdot_affine(ac, bc) ≈ cdot_stride(ac, bc)
        

        xq = [ntuple(_ -> rand(), Val(4)) for _ ∈ 1:i];
        yq = [ntuple(_ -> rand(), Val(4)) for _ ∈ 1:i];
        xqv = reinterpret(Float64, xq);
        yqv = reinterpret(Float64, yq);
        qsimd = Base.vect(qdot_simd(xq, yq)...);
        if VERSION ≥ v"1.6.0-rc1"
            xqm = reinterpret(reshape, Float64, xq);
            yqm = reinterpret(reshape, Float64, yq);
            @test qsimd ≈ Base.vect(qdot_mat(xqm, yqm)...)
        end
        @test qsimd ≈ Base.vect(qdot_affine(xqv, yqv)...) ≈ Base.vect(qdot_stride(xqv, yqv)...)

        if VERSION ≥ v"1.6.0-rc1"
            Ac = rand(Complex{Float64}, i, i);
            Bc = rand(Complex{Float64}, i, i);
            Cc1 = Ac*Bc;
            Cc2 = similar(Cc1);
            # Cc3 = similar(Cc1)
            Cca = reinterpret(reshape, Float64, Cc2);
            Aca = reinterpret(reshape, Float64, Ac);
            Bca = reinterpret(reshape, Float64, Bc);
            cmatmul_array!(Cca, Aca, Bca)
            
            @test Cc1 ≈ Cc2# ≈ Cc3
        end
    end
    @show @__LINE__
    if VERSION ≥ v"1.6.0-rc1"
        M = 10
        G = 50
        J = 50
        H = 30

        # B = rand(Complex{Float64}, 2*J+1, G-1, H+1, M+1);
        # ϕ = rand(Complex{Float64}, 2*J+1, G+1, H+1, M+1);
        rbc = let rb = 1.0:((2*J+17) * (G+15) * (H+17) * (M+17)), rbr = reverse(rb)
          Complex{Float64}[rb[i] + im * rbr[i] for i ∈ eachindex(rb)];
        end
        B = view(reshape(rbc, (2*J+17, G+15, H+17, M+17)), 9:2*J+9, 9:G+9, 9:H+9, 9:M+9) .= rand.() .+ rand.().*im;
        ϕ = view(fill(1e5+1e7im, 2*J+17, G+17, H+17, M+17), 9:2*J+9, 9:G+9, 9:H+9, 9:M+9) .= rand.() .+ rand.().*im;
        @test issue209(M, G, J, H, B, ϕ) ≈ issue209_noavx(M, G, J, H, B, ϕ)
    end
end


