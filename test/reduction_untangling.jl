
# issue 288
function not_a_reduction!(A, B)
  @turbo for j ∈ 1:size(A.re, 2)
    jre = B.re[j]
    jim = B.im[j]
    for i ∈ 1:size(A.re, 1)
      ire = B.re[i]
      iim = B.im[i]
      cisim = iim * jre - ire * jim
      cisre = ire * jre + iim * jim
      ρre_i = A.re[i,j]
      ρim_i = A.im[i,j]
      re_out = ρre_i * cisre - ρim_i * cisim
      im_out = ρre_i * cisim + ρim_i * cisre
      A.re[i,j] = re_out
      A.im[i,j] = im_out
    end
  end
  return nothing
end
function not_a_reduction_noturbo!(A, B)
  @turbo for j ∈ 1:size(A.re, 2)
    jre = B.re[j]
    jim = B.im[j]
    for i ∈ 1:size(A.re, 1)
      ire = B.re[i]
      iim = B.im[i]
      cisim = iim * jre - ire * jim
      cisre = ire * jre + iim * jim
      ρre_i = A.re[i,j]
      ρim_i = A.im[i,j]
      re_out = ρre_i * cisre - ρim_i * cisim
      im_out = ρre_i * cisim + ρim_i * cisre
      A.re[i,j] = re_out
      A.im[i,j] = im_out
    end
  end
  return nothing
end

@testset "Untangle reductions" begin
  @show @__LINE__
  N = 11
  A1 = (re = rand(N,N), im = rand(N,N))
  A2 = deepcopy(A1)
  B = (re = rand(N), im = rand(N))
  not_a_reduction!(A1, B)
  not_a_reduction_noturbo!(A2, B)
  @test A1.re ≈ A2.re
  @test A1.im ≈ A2.im
end

