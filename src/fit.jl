abstract type AbstractOptimizationMethod end

struct Nesterov <: AbstractOptimizationMethod
    α₀::Real # Initial step size
    restarts::Int # Restart step size

    function Nesterov()
        α₀ = 1
        restarts = 50
        new(α₀, restarts)
    end
end

struct GD <: AbstractOptimizationMethod
    α₀::Real # Initial step size
    α::Function # Step size

    function GD(; α₀::Real = 1)
        α(r::Int; α₀=α₀) = α₀/r
        new(α₀, α)
    end
end

# function fit!(
#     loss::AbstractLoss,
#     GD::GD,
#     triplets::Triplets,
#     X::AbstractEmbedding;
#     verbose::Bool = true,
#     max_iterations::Int = 1000,
#     info_iterations::Int = 10
# )

#     @assert max_iterations >= 10 "Iterations must be at least 10"
#     # @assert maximum([maximum(t) for t in triplets]) == nitems(X)
    
#     C = Inf                           # Cost
#     ∇C = Inf * ones(Float64, size(X)) # Gradient

#     tolerance = 1e-7 # convergence tolerance

#     # Perform main iterations
#     r = 0   # Number of iterations (so far)
#     pviolations = 0.0 # Percentage of violations

#     while r < max_iterations && !isapprox(norm(∇C), 0.0, atol=tolerance)

#         r += 1

#         # Calculate gradient descent and cost
#         C, ∇C = gradient(loss, triplets, X)

#         # Update the embedding according to the gradient
#         X.X = X.X .- GD.α(r) * ∇C
#         # X.X = X.X .- 1/1000 * ∇C
#         X.X = X.X/maximum(X.X)

#         # Print out progress
#         if verbose && (r % info_iterations == 0)
#             # If we have a large number of triplets, computing the number of violations
#             # can be costly. Therefore, we only perform this operation every info_iterations iterations
#             # If more details are needed, you can set the environment variable
#             # JULIA_DEBUG=TripletEmbeddings before starting Julia.
#             pviolations = percent_violations(triplets, X)
#             @info @sprintf "loss = %s, iteration = %d, cost = %.2f, violations = %.2f%%, ||∇C|| = %.2f" typeof(loss) r C 100 * pviolations norm(∇C)
#         end

#         @debug "Iteration = $r, Cost = $C, nincrements = $nincrements" X.X percent_violations(triplets, X)
#     end

#     if !verbose
#         pviolations = percent_violations(triplets, X)
#     end
    
#     return pviolations
# end


# function fit!(
#     loss::AbstractLoss,
#     method::Nesterov,
#     triplets::Triplets,
#     X::AbstractEmbedding;
#     verbose::Bool = true,
#     max_iterations::Int = 1000
#     )

#     @assert max_iterations >= 10 "Iterations must be at least 10"

#     C = Inf                      # Cost
#     ∇C = Inf * ones(Float64, size(X)) # Gradient

#     d, n = size(X)

#     y = [Embedding(ndims(X), nitems(X)) for _ in  1:max_iterations)]
#     x = [Embedding(ndims(X), nitems(X)) for _ in  1:max_iterations)]

#     tolerance = 1e-7

#     α = zeros(Float64, max_iterations)
#     t = zeros(Float64, max_iterations)
#     r = 2 # Iterations

#     while r < max_iterations && !isapprox(norm(∇C), zeros(d, n), atol=tolerance)
        
#         C, ∇C = gradient(loss, triplets, x[r])

#         r += 1
#         α[r] = (1 + √(4 * α[r-1]^2 + 1)) / 2
#         t[r-1] = (α[r-1] - 1) / α[r]

#         y[r] = (1 + t[r-1]) * x[:,r-1] - t[r-1] * x[r-2]
#         x[r] = y[:,r] - method.α₀/L * ∇f(y[:,r])

#         !(verbose && (r % 100 == 0)) || @printf "r = %g, f(x) = %f, ||∇f(x)|| = %f\n" r f(x[:,r]) norm(∇f(x[:,r]))

#     end
# end

function fit!(
    loss::AbstractLoss,
    triplets::Triplets,
    X::AbstractEmbedding;
    verbose::Bool = true,
    max_iterations::Int64 = 1000
)

    @assert max_iterations >= 10 "Iterations must be at least 10"
    # @assert maximum([maximum(t) for t in triplets]) == nitems(X)
    
    C = Inf                      # Cost
    ∇C = zeros(Float64, size(X)) # Gradient

    tolerance = 1e-7 # convergence tolerance
    η = 1.0          # learning rate
    best_C = Inf     # best error obtained so far
    best_X = X       # best embedding found so far

    # Perform main iterations
    niterations = 0   # Number of iterations (so far)
    nincrements = 0   # Number of increments
    pviolations = 0.0 # Percentage of violations

    while niterations < max_iterations && nincrements < 5
        niterations += 1
        old_C = C

        # Calculate gradient descent and cost
        C, ∇C = gradient(loss, triplets, X)

        # Update the embedding according to the gradient
        X.X = X.X .- (η / ntriplets(triplets) * nitems(X)) * ∇C
        # X.X = X.X/norm(X.X)

        if C < best_C
            best_C = C
            best_X = X
        end

        # Update learning rate
        if old_C > C + tolerance
            nincrements = 0
            η *= 1.01
        else
            nincrements += 1
            η *= 0.5
        end

        # Print out progress
        if verbose && (niterations % 10 == 0)
            # If we have a large number of triplets, computing the number of violations
            # can be costly. Therefore, we only perform this operation every 10 iterations
            # If more details are needed, you can set the environment variable
            # JULIA_DEBUG=TripletEmbeddings before starting Julia.
            pviolations = percent_violations(triplets, X)
            @info @sprintf "loss = %s, iteration = %d, cost = %.2f, violations = %.2f%%, ||∇L|| = %.2f" typeof(loss) niterations C 100 * pviolations norm(∇C)
        end

        @debug "Iteration = $niterations, Cost = $C, nincrements = $nincrements" X.X percent_violations(triplets, X)
    end

    if !verbose
        pviolations = percent_violations(triplets, X)
    end
    
    X = best_X
    return pviolations
end

function percent_violations(triplets::Triplets, X::AbstractEmbedding)
    D = pairwise(SqEuclidean(), X, dims=2)

    nviolations = 0

    for t = 1:ntriplets(triplets)
        nviolations += D[triplets[t][1], triplets[t][2]] > D[triplets[t][1], triplets[t][3]]
    end

    return nviolations/ntriplets(triplets)
end