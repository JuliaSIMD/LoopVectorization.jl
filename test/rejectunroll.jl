using LoopVectorization, Test

const GAMMA = 1.4
@testset "Reject Unroll" begin
  function r_manual_inline!(r, u)
    @inbounds @fastmath begin
      m = size(u, 2)
      n = size(u, 3)
      fill!(r, 0.0)
      for j = 2:n-1
        for i = 1:m-1
          rl = u[1, i, j]
          rul = u[2, i, j]
          rvl = u[3, i, j]
          rEl = u[4, i, j]

          rr = u[1, i+1, j]
          rur = u[2, i+1, j]
          rvr = u[3, i+1, j]
          rEr = u[4, i+1, j]

          rr1 = 1.0 / rr
          ur = rur * rr1
          vr = rvr * rr1
          Er = rEr * rr1
          u2r = ur * ur + vr * vr
          pr = (GAMMA - 1.0) * (rEr - 0.5 * rr * u2r)
          hr = Er + pr * rr1
          unr = ur

          rhr = rEr + pr

          rl1 = 1.0 / rl
          ul = rul * rl1
          vl = rvl * rl1
          El = rEl * rl1
          u2l = ul * ul + vl * vl
          pl = (GAMMA - 1.0) * (rEl - 0.5 * rl * u2l)
          hl = El + pl * rl1
          unl = ul

          rhl = rEl + pl

          di = sqrt(rr * rl1)
          d1 = 1.0 / (di + 1.0)
          ui = (di * ur + ul) * d1
          vi = (di * vr + vl) * d1
          hi = (di * hr + hl) * d1
          ci2 = (GAMMA - 1.0) * (hi - 0.5 * (ui * ui + vi * vi))
          ci = sqrt(ci2)
          af = 0.5 * (ui * ui + vi * vi)
          uni = ui

          dr = rr - rl
          dru = rur - rul
          drv = rvr - rvl
          drE = rEr - rEl

          rlam1 = abs(uni + ci)
          rlam2 = abs(uni - ci)
          rlam3 = abs(uni)

          s1 = 0.5 * (rlam1 + rlam2)
          s2 = 0.5 * (rlam1 - rlam2)
          al1x = (GAMMA - 1.0) * (af * dr - ui * dru - vi * drv + drE)
          al2x = -uni * dr + dru
          cc1 = ((s1 - rlam3) * al1x / ci2) + (s2 * al2x / ci)
          cc2 = (s2 * al1x / ci) + (s1 - rlam3) * al2x

          f1 = 0.5 * (rr * unr + rl * unl) - 0.5 * (rlam3 * dr + cc1)
          f2 =
            0.5 * (rur * unr + rul * unl) + 0.5 * (pr + pl) -
            0.5 * (rlam3 * dru + cc1 * ui + cc2)
          f3 = 0.5 * (rvr * unr + rvl * unl) - 0.5 * (rlam3 * drv + cc1 * vi)
          f4 = 0.5 * (rhr * unr + rhl * unl) - 0.5 * (rlam3 * drE + cc1 * hi + cc2 * uni)

          r[1, i, j] -= f1
          r[2, i, j] -= f2
          r[3, i, j] -= f3
          r[4, i, j] -= f4

          r[1, i+1, j] += f1
          r[2, i+1, j] += f2
          r[3, i+1, j] += f3
          r[4, i+1, j] += f4
        end
      end
    end
    r
  end

  function r_turbo!(r, u)
    @inbounds begin
      m = size(u, 2)
      n = size(u, 3)
      fill!(r, 0.0)
      @turbo for j = 2:n-1
        for i = 1:m-1
          rl = u[1, i, j]
          rul = u[2, i, j]
          rvl = u[3, i, j]
          rEl = u[4, i, j]

          rr = u[1, i+1, j]
          rur = u[2, i+1, j]
          rvr = u[3, i+1, j]
          rEr = u[4, i+1, j]

          rr1 = 1.0 / rr
          ur = rur * rr1
          vr = rvr * rr1
          Er = rEr * rr1
          u2r = ur * ur + vr * vr
          pr = (GAMMA - 1.0) * (rEr - 0.5 * rr * u2r)
          hr = Er + pr * rr1
          unr = ur

          rhr = rEr + pr

          rl1 = 1.0 / rl
          ul = rul * rl1
          vl = rvl * rl1
          El = rEl * rl1
          u2l = ul * ul + vl * vl
          pl = (GAMMA - 1.0) * (rEl - 0.5 * rl * u2l)
          hl = El + pl * rl1
          unl = ul

          rhl = rEl + pl

          di = sqrt(rr * rl1)
          d1 = 1.0 / (di + 1.0)
          ui = (di * ur + ul) * d1
          vi = (di * vr + vl) * d1
          hi = (di * hr + hl) * d1
          ci2 = (GAMMA - 1.0) * (hi - 0.5 * (ui * ui + vi * vi))
          ci = sqrt(ci2)
          af = 0.5 * (ui * ui + vi * vi)
          uni = ui

          dr = rr - rl
          dru = rur - rul
          drv = rvr - rvl
          drE = rEr - rEl

          rlam1 = abs(uni + ci)
          rlam2 = abs(uni - ci)
          rlam3 = abs(uni)

          s1 = 0.5 * (rlam1 + rlam2)
          s2 = 0.5 * (rlam1 - rlam2)
          al1x = (GAMMA - 1.0) * (af * dr - ui * dru - vi * drv + drE)
          al2x = -uni * dr + dru
          cc1 = ((s1 - rlam3) * al1x / ci2) + (s2 * al2x / ci)
          cc2 = (s2 * al1x / ci) + (s1 - rlam3) * al2x

          f1 = 0.5 * (rr * unr + rl * unl) - 0.5 * (rlam3 * dr + cc1)
          f2 =
            0.5 * (rur * unr + rul * unl) + 0.5 * (pr + pl) -
            0.5 * (rlam3 * dru + cc1 * ui + cc2)
          f3 = 0.5 * (rvr * unr + rvl * unl) - 0.5 * (rlam3 * drv + cc1 * vi)
          f4 = 0.5 * (rhr * unr + rhl * unl) - 0.5 * (rlam3 * drE + cc1 * hi + cc2 * uni)

          r[1, i, j] -= f1
          r[2, i, j] -= f2
          r[3, i, j] -= f3
          r[4, i, j] -= f4

          r[1, i+1, j] += f1
          r[2, i+1, j] += f2
          r[3, i+1, j] += f3
          r[4, i+1, j] += f4
        end
      end
    end
    r
  end

  u = rand(4, 128, 128)
  u[[1, 4], :, :] .= 1.0
  u[[2, 3], :, :] .= 0.1
  @test r_manual_inline!(similar(u), u) ≈ r_turbo!(similar(u), u)

end
