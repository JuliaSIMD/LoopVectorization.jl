using LoopVectorization, Test, ArrayInterface

# macros for generate loops whose body is not a block
macro gen_loop_issue395(ex)
  sym, ind = ex.args
  loop_body = :(ret[$ind] = $sym[$ind])
  loop = Expr(:for, :($ind = axes($sym, 1)), loop_body)
  return esc(:(@turbo $loop))
end
macro gen_single_loop(B, A)
  loop_body = :($B[i] = $A[i])
  loop = Expr(:for, :(i = indices(($B, $A), 1)), loop_body)
  return esc(:(@turbo $loop))
end
macro gen_nest_loop(C, A, B)
  loop_body = :($C[i, j] = $A[i] * $B[j])
  loop_head = Expr(:block, :(j = indices(($C, $B), (2, 1))), :(i = indices(($C, $A), 1)))
  loop = Expr(:for, loop_head, loop_body)
  return esc(:(@turbo $loop))
end
macro gen_A_mul_B(C, A, B)
  inner_body = :(Cji += $A[j, k] * $B[k, i])
  inner_loop = Expr(:for, :(k = indices(($A, $B), (2, 1))), inner_body)
  loop = :(
    for i in indices(($C, $B), 2), j in indices(($C, $A), 1)
      Cji = zero(eltype($C))
      $inner_loop
      $C[j, i] = Cji
    end
  )
  return esc(:(@turbo $loop))
end

@testset "check_block, #395" begin
  A = rand(4)
  B = rand(4)
  C = rand(4, 4)
  D = zeros(4)
  E = zeros(4, 4)
  F = zeros(4, 4)
  ret = zeros(4)
  @gen_single_loop D A
  @gen_loop_issue395 B[i]
  @gen_nest_loop E A B
  @gen_A_mul_B F C E
  @test D == A
  @test ret == B
  @test E == A * B'
  @test F == C * E
end

@testset "enumerate, #393" begin
  A = zeros(4)
  B = zeros(4)
  C = zeros(4)
  @turbo for (i, x) in enumerate(A)
    A[i] = i + x
  end
  @turbo for (i,) in enumerate(B)
    B[i] += 1
  end
  @turbo for ix in enumerate(C)
    C[ix[1]] = ix[1] + ix[2]
  end
  @test_throws ArgumentError @turbo for () in enumerate(A) end
  @test A == 1:4
  @test B == 1:4
  @test C == 1:4
end
