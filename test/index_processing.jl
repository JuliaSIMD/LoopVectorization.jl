

function multiindexreverse(n, m)
  A = zeros(n, 2m)
  @turbo for i = 1:n÷2, j = 1:m
    a = 1.23
    b = 3.21
    A[i, j] = a
    A[i, m+j] = b
    r = n + 1 - i
    A[r, j] = a
    A[r, m+j] = b
  end
  return A
end
function multiindexreverse_ref(n, m)
  A = zeros(n, 2m)
  @turbo for i = 1:n÷2, j = 1:m
    a = 1.23
    b = 3.21
    A[i, j] = a
    A[i, m+j] = b
    r = n + 1 - i
    A[r, j] = a
    A[r, m+j] = b
  end
  return A
end

@testset "Multiple indices" begin
  for m = 1:10, n = 1:20
    @test multiindexreverse(n, m) == multiindexreverse_ref(n, m)
  end
end
