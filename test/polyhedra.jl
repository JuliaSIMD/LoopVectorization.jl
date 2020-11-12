
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
# for m ∈ 1:M, n ∈ 1:m
p0 = LoopVectorization.Polyhedra(
    ((ByteVector(0,0),ByteVector(0,0)), # Aₗ
     (ByteVector(0,0),ByteVector(1,0))), # Aᵤ
    LoopVectorization.RectangularPolyhedra(
        ((1,1),(0,0)),
        ((0,0),(-1024,0)),
        (ByteVector(),ByteVector(1))
    )
);

LoopVectorization.getloop(p0, ByteVector(1,2), LoopVectorization.VectorLength(8), 1, 1024)
LoopVectorization.getloop(p0, ByteVector(1), LoopVectorization.VectorLength(8), 1, 1)

# for m ∈ 1:M, n ∈ m:N, k ∈ 1:m+n+K
p1 = Polyhedra(
    ((),
     ()),
    RectangularPolyhedra(

    )
)


