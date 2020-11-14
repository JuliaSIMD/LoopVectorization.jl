
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

using LoopVectorization: ByteVector

function sumloop0(M, W, i)
    r1 = i == 1 ? (n = cld(M,W); M-W*n:W:M) : 1:1:M
    W2 = i == 2 ? W : 1
    s = 0
    for m ∈ r1, n ∈ 1:W2:m
        s += 1
    end
    s
end

# for m ∈ 1:M, n ∈ 1:m
function polyloop0(M, W, i, order)
    p0 = LoopVectorization.Polyhedra(
        ((ByteVector(0,0),ByteVector(0,0)), # Aₗ
         (ByteVector(0,0),ByteVector(1,0))), # Aᵤ
        LoopVectorization.RectangularPolyhedra(
            ((1,1),(0,0)), # cₗ, cᵤ
            ((0,0),(-M,0)), # dₗ, dᵤ
            (ByteVector(),ByteVector(1))
        )
    );
    first(LoopVectorization.getloop(p0, order, LoopVectorization.VectorLength(W), i, M))
end

for i ∈ 0:2, W ∈ (1, 4, 8, 16), M ∈ 1:128
    @test sumloop0(M, W, i) == polyloop0(M, W, i, ByteVector(1,2))
end

LoopVectorization.getloop(p0, ByteVector(1,2), LoopVectorization.VectorLength(8), 1, 1024)
LoopVectorization.getloop(p0, ByteVector(1), LoopVectorization.VectorLength(8), 1, 1)

# for m ∈ 1:M, n ∈ 1:m
p1 = LoopVectorization.Polyhedra(
    ((ByteVector(0,0),ByteVector(-1,0)), # Aₗ
     (ByteVector(0,0),ByteVector(0,0))), # Aᵤ
    LoopVectorization.RectangularPolyhedra(
        ((1,0),(0,0)), # cₗ, cᵤ
        ((0,0),(-1024,-1024)), # dₗ, dᵤ
        (ByteVector(),ByteVector(1))
    )
);

LoopVectorization.getloop(p0, ByteVector(1,2), LoopVectorization.VectorLength(8), 1, 1024)
LoopVectorization.getloop(p0, ByteVector(1), LoopVectorization.VectorLength(8), 1, 1)

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


