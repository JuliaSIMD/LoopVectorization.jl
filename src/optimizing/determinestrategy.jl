

function determine_cost_statement()

end

# special case for when we only have one statement
function determine_cost_looporder_1_vector(ls::LoopSet, statement, remaining_loops, v, u₁, u₂)

    
end

# costs should be based on
# NTuple{4,Float64} # costs # reduced by neither, reduced by u₁, reduced by u₂, reduced by both
# NTuple{4,Float64} # reg pressure: reg remaining, 1 per u₁, 1 per u₂, 1 per u₁ * u₂
# reg remaining is minus for normal u₂ and minus for things depending on neither
#
# TODO add fields indicating unrolled loop sizes, and whether they're triangular
# How to handle vectorized loop with respect to triangles?
struct CostSummary
    costs::NTuple{4,Float64}
    reg_pres::NTuple{4,Float64}
end
function CostSummary
    costs = (0.0,0.0,0.0,0.0)
    reg_pres = (0.0,0.0,0.0,0.0)
    CostSummary(costs, reg_pres)
end

# remaining loops vector with element per separable statement, each element a vector of indices for remaining loops
function determine_cost_looporder(ls::LoopSet, separable_statements, remaining_loops, v, u₁, u₂, cost_summary)
    # isvalid() || return Inf
    n_statements = length(separable_statements)
    nest_depth = ByteVector{UInt64}()
    for loopnest ∈ remaining_loops
        nest_depth = push(nest_depth, length(loopnest))
    end
    for combination ∈ DynamicCartesian(nest_depth)
        for fusion_flag ∈ 0x00:((0x01 << (n_statements - 1)) - 0x01)
            # check legality
            ff = fusion_flag
            num_iterations = 0 # < 0 means not a compatible combination
            for n ∈ 2:n_statements
                if ff % Bool
                    num_iterations = check_loop_compatibility(
                        separable_statements[n-1], remaining_loops[n-1][combination[n-1]],
                        separable_statements[n],   remaining_loops[n][combination[n]],
                        num_iterations
                    )
                    num_iterations < 0 && break
                end
                ff >>>= 1
            end
            num_iterations < 0 && continue
            for u₂ᵢ ∈ false:u₂
                for vᵢ ∈ false:v
                    for u₁ᵢ ∈ false:u₁
                        
                    end
                end
            end
        end
    end
    
    bestcost = Inf
    local bests::Schedule
    bestu₁ = bestu₂ = bestv = false
    
    for i ∈ eachindex(remainingloops)
        for u₂ᵢ ∈ false:u₂

            
            for vᵢ ∈ false:v, u₁ᵢ ∈ false:u₁, 
                statementsᵢ = 0
                s, c = determine_cost(ls, separable_statements, remaining_loops, v, u₁, u₂)
                if c < bestcost
                    bestu₁ = u₁ᵢ
                    bestu₂ = u₂ᵢ
                    bestv = vᵢ
                    bests = s
                    bestcost = c
                end
            end
        end
    end
    Schedule(statements, schedules, bestv, bestu₁, bestu₂)
end



@generated function permiter(::Val{N}) where {N}
    quote
        s = Base.Cartesian.@ntuple $N i -> 0
        tup_0 = ntuple(identity, Val{$N}())
        @inbounds Base.Cartesian.@nloops $N i n -> Base.OneTo(n) n -> begin
            tup_{$(N+1)-n} = ArrayInterface.deleteat(tup_{$N-n}, i_n)
            p_{$(N+1)-n} = tup_{$N-n}[i_n]
        end begin
            s = s .+ Base.Cartesian.@ntuple $N p
            # @show
        end
        s
    end
end



