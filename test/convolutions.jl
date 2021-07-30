
function compute_update0!(H2, dHdtau, H, min_dxy2, _dx, _dy, dx, dy)
  @inbounds @fastmath for iy=1:size(dHdtau,2)
    for ix=1:size(dHdtau,1)
      dHdtau[ix, iy] = (-((-((H[(ix + 1) + 1, iy + 1] - H[ix + 1, iy + 1])) * _dx - -((H[ix + 1, iy + 1] - H[ix, iy + 1])) * _dx)) * _dx - (-((H[ix + 1, (iy + 1) + 1] - H[ix + 1, iy + 1])) * _dy - -((H[ix + 1, iy + 1] - H[ix + 1, iy])) * _dy) * _dy) + 0.6 * dHdtau[ix, iy]
      H2[ix+1,iy+1] = H[ix+1,iy+1] + min_dxy2/4.1*dHdtau[ix,iy]  # sets the BC as H[1]=H[end]=0
    end
  end
  return
end
function compute_update1!(H2, dHdtau, H, min_dxy2, _dx, _dy, dx, dy)
  @turbo for iy=1:size(dHdtau,2)
    for ix=1:size(dHdtau,1)
      dHdtau[ix, iy] = (-((-((H[(ix + 1) + 1, iy + 1] - H[ix + 1, iy + 1])) * _dx - -((H[ix + 1, iy + 1] - H[ix, iy + 1])) * _dx)) * _dx - (-((H[ix + 1, (iy + 1) + 1] - H[ix + 1, iy + 1])) * _dy - -((H[ix + 1, iy + 1] - H[ix + 1, iy])) * _dy) * _dy) + 0.6 * dHdtau[ix, iy]
      H2[ix+1,iy+1] = H[ix+1,iy+1] + min_dxy2/4.1*dHdtau[ix,iy]  # sets the BC as H[1]=H[end]=0
    end
  end
  return
end

@testset "conv" begin
  lx, ly = 10.0, 10.0   # domain size
  nx, ny = 12, 12     # number of grid points
  nt     = 1e3          # max number of iterations
  dx, dy = lx/nx, ly/ny
  xc, yc = LinRange(dx/2, lx-dx/2, nx), LinRange(dy/2, ly-dy/2, ny)
  dHdtau = zeros(nx-2, ny-2)
  dHdtau2 = zeros(nx-2, ny-2)
  H      = exp.(.-(xc.-lx/2).^2 .-(yc.-ly/2)'.^2)
  H1     = copy(H)
  H2     = copy(H)
  min_dxy2 = min(dx,dy)^2
  _dx, _dy = 1.0/dx, 1.0/dy
  compute_update0!(H1, dHdtau, H, min_dxy2, _dx, _dy, dx, dy)
  compute_update1!(H2, dHdtau2, H, min_dxy2, _dx, _dy, dx, dy)
  @test H1 ≈ H2
  @test dHdtau ≈ dHdtau2
end
