
# p1 = Polyhedra(
#     [  1   0   0   0   0   0;
#        -1  0   0   1   0   0;
#        0   1   0   0   0   0;
#        0  -1   0   0   1   0;
#        -1  0   1   0   0   0;
#        0   0  -1   0   0   1],
#     [1, 0, 1, 0, 1, 0], # b
#     Float64[1024, 1024, 1024], # parameters
#     [1, 2, 3] # ids of parameters with respect to ref in LoopSet; negative numbers mean they're a loop
# )

using LoopVectorization, Test
using LoopVectorization: ByteVector

function sumloop0(M, W, i, l1 = 1, l2 = 1)
    r1 = i == 1 ? (n = cld(M - l1 + 1, W)-1; M-W*n:W:M) : l1:1:M
    W2 = i == 2 ? W : 1
    s = 0
    for m ∈ r1, n ∈ l2:W2:m
        s += 1
    end
    s
end

# for m ∈ 1:M, n ∈ 1:m
function polyloop0(M, W, i, order, l1 = 1, l2 = 1, citers = 1)
    p0 = LoopVectorization.Polyhedra(
        ((ByteVector(0,0),ByteVector(0,0)), # Aₗ
         (ByteVector(0,0),ByteVector(1,0))), # Aᵤ
        LoopVectorization.RectangularPolyhedra(
            ((l1,l2),(0,0)), # cₗ, cᵤ
            ((0,0),(-M,0)), # dₗ, dᵤ
            (ByteVector(),ByteVector(1))
        )
    );
    first(LoopVectorization.getloop(p0, order, LoopVectorization.VectorLength(W), i, citers))
end

for i ∈ 0:2, W ∈ (1, 4, 8, 16), M ∈ -48:48, l1 ∈ -53:M, l2 ∈ -53:l1
    s = sumloop0(M, W, i, l1, l2)
    @test s == polyloop0(M, W, i, ByteVector(1,2), l1, l2, M)
    @test s == polyloop0(M, W, i, ByteVector(2,1), l1, l2, M)
    @test (i == 1 ? cld(M+1-l1,W) : M+1-l1) == polyloop0(M, W, i, ByteVector(1), l1, l2, 1)
    @test (i == 2 ? cld(M+1-l2,W) : M+1-l2) == polyloop0(M, W, i, ByteVector(2), l1, l2, 1)
end

LoopVectorization.getloop(p0, ByteVector(1,2), LoopVectorization.VectorLength(8), 1, 1024)
LoopVectorization.getloop(p0, ByteVector(1), LoopVectorization.VectorLength(8), 1, 1)

# for m ∈ 1:M, n ∈ 1:m
function polyloop1(M, W, i, order, l1 = 1, N = M, citers = 1)
    p1 = LoopVectorization.Polyhedra(
        ((ByteVector(0,0),ByteVector(-1,0)), # Aₗ
         (ByteVector(0,0),ByteVector(0,0))), # Aᵤ
        LoopVectorization.RectangularPolyhedra(
            ((l1,0),(0,0)), # cₗ, cᵤ
            ((0,0),(-M,-N)), # dₗ, dᵤ
            (ByteVector(),ByteVector(1))
        )
    );
    first(LoopVectorization.getloop(p1, order, LoopVectorization.VectorLength(W), i, citers))
end
for i ∈ 0:2, W ∈ (1, 4, 8, 16), M ∈ 1:128
    s = sumloop0(M, W, i)
    @test s == polyloop1(M, W, i, ByteVector(1,2), 1, M)
    @test s == polyloop1(M, W, i, ByteVector(2,1), 1, M)
    @test (i == 1 ? cld(M,W) : M) == polyloop1(M, W, i, ByteVector(1), 1, M)
    @test (i == 2 ? cld(M,W) : M) == polyloop1(M, W, i, ByteVector(2), 1, M)
end

LoopVectorization.getloop(p1, ByteVector(1,2), LoopVectorization.VectorLength(8), 1, 1024)
LoopVectorization.getloop(p1, ByteVector(1), LoopVectorization.VectorLength(8), 1, 1)

# for m ∈ 1:M, n ∈ m:N, k ∈ 1:m+n+K
p2 = LoopVectorization.Polyhedra(
    ((ByteVector(0,0,0),ByteVector(0,0)), # Aₗ
     (ByteVector(0,0,0),ByteVector(1,0))), # Aᵤ
    LoopVectorization.RectangularPolyhedra(
        ((1,1),(0,0)), # cₗ, cᵤ
        ((0,0),(-1024,0)), # dₗ, dᵤ
        (ByteVector(),ByteVector(1))
    )
);


