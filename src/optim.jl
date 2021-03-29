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
#     X::Embedding;
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
#     X::Embedding;
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