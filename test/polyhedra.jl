
p1 = Polyhedra(
    [  1   0   0   0   0   0;
       -1  0   0   1   0   0;
       0   1   0   0   0   0;
       0  -1   0   0   1   0;
       -1  0   1   0   0   0;
       0   0  -1   0   0   1],
    [1, 0, 1, 0, 1, 0], # b
    Float64[1024, 1024, 1024], # parameters
    [1, 2, 3] # ids of parameters with respect to ref in LoopSet; negative numbers mean they're a loop
)


