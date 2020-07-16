

function contract!(tiJaB_d_temp3, tiJaB_i, Wmnij)
    rvir = axes(tiJaB_d_temp3, 4)
    nvir = last(rvir)
    rocc = axes(tiJaB_d_temp3, 1)
    @inbounds @fastmath for b in rvir, a in b:nvir, j in rocc, i in j:last(rocc)
        temp = zero(eltype(tiJaB_i))
        for n in rocc, m in rocc
            temp += tiJaB_i[m,n,a,b]*Wmnij[m,n,i,j]
        end
        tiJaB_d_temp3[i,j,a,b] = temp
        tiJaB_d_temp3[j,i,b,a] = temp
    end
end

function contracttest1!(tiJaB_d_temp3, tiJaB_i, Wmnij)
    rvir = axes(tiJaB_d_temp3, 4)
    nvir = last(rvir)
    rocc = axes(tiJaB_d_temp3, 1)
    for b in rvir, j in rocc
        @avx for a in b:nvir, i in j:last(rocc)
            temp = zero(eltype(tiJaB_i))
            for n in rocc, m in rocc
                temp += tiJaB_i[m,n,a,b]*Wmnij[m,n,i,j]
            end
            tiJaB_d_temp3[i,j,a,b] = temp
            tiJaB_d_temp3[j,i,b,a] = temp
        end
    end
    tiJaB_d_temp3
end
function contracttest2!(tiJaB_d_temp3, tiJaB_i, Wmnij)
    rvir = axes(tiJaB_d_temp3, 4)
    nvir = last(rvir)
    rocc = axes(tiJaB_d_temp3, 1)
    for b in rvir, a in b:nvir, j in rocc
        @avx for i in j:last(rocc)
            temp = zero(eltype(tiJaB_i))
            for n in rocc, m in rocc
                temp += tiJaB_i[m,n,a,b]*Wmnij[m,n,i,j]
            end
            tiJaB_d_temp3[i,j,a,b] = temp
            tiJaB_d_temp3[j,i,b,a] = temp
        end
    end
    tiJaB_d_temp3
end

@testset "Tensors" begin
    @show Float64, @__LINE__
    LA, LIM = 31, 23;
    A = rand(LIM, LIM, LA, LA);
    B = rand(LIM, LIM, LIM, LIM);

    C1 = fill(-999.999, LIM, LIM, LA, LA);
    C2 = fill(-999.999, LIM, LIM, LA, LA);
    C3 = fill(-999.999, LIM, LIM, LA, LA);
    # C1 = Array{Float64}(undef, LIM, LIM, LA, LA);
    # C2 = similar(C1); C3 = similar(C1);

    contract!(C1, A, B)
    @test C1 ≈ contracttest1!(C2, A, B)
    @test C1 ≈ contracttest2!(C3, A, B)

    Apermute = PermutedDimsArray(permutedims(A, (2,4,1,3)), (3,1,4,2));
    Bpermute = PermutedDimsArray(permutedims(B, (2,1,4,3)), (2,1,4,3));
    @test C1 ≈ contracttest1!(C2, Apermute, Bpermute)
    @test C1 ≈ contracttest2!(C3, Apermute, Bpermute)
end


