function fit!(
    loss::AbstractLoss,
    triplets,
    X::Embedding;
    verbose::Bool = true,
    print_every::Int = 10,
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
        X.X = X.X .- (η / length(triplets) * nitems(X)) * ∇C
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
        if verbose && (niterations % print_every == 0)
            # If we have a large number of triplets, computing the number of violations
            # can be costly. Therefore, we only perform this operation every 10 iterations
            # If more details are needed, you can set the environment variable
            # JULIA_DEBUG=TripletEmbeddings before starting Julia.
            pviolations = percent_violations(triplets, X)
            @info @sprintf "loss = %s, iteration = %d, cost = %.2f, misclassifications = %.2f%%, ||∇ₓR|| = %.2f" typeof(loss) niterations C/size(triplets)[1] 100 * pviolations norm(∇C)
        end

        @debug "Iteration = $niterations, Cost = $C, nincrements = $nincrements" X.X percent_violations(triplets, X)
    end

    if !verbose
        pviolations = percent_violations(triplets, X)
    end

    X = best_X
    return pviolations
end

function percent_violations(triplets::Triplets, X::Embedding)
    D = pairwise(SqEuclidean(), X.X, dims=2)

    nviolations = 0

    for t in triplets
        nviolations += D[t[:i], t[:j]] > D[t[:i], t[:k]]
    end

    return nviolations/length(triplets)
end

# function fit!(
#         loss::uSTE,
#         triplets::LabeledTriplets,
#         X::Embedding;
#         verbose::Bool = true,
#         print_every::Int = 10,
#         max_iterations::Int64 = 1000
#     )

#     # @assert max_iterations >= 10 "Iterations must be at least 10"

#     C = Inf                      # Cost
#     ∇C = zeros(Float64, size(X)) # Gradient

#     tolerance = 1e-7 # convergence tolerance
#     η = 1.0          # learning rate
#     best_C = Inf     # best error obtained so far
#     best_X = X       # best embedding found so far

#     # Perform main iterations
#     niterations = 0   # Number of iterations (so far)
#     nincrements = 0   # Number of increments
#     pviolations = 0.0 # Percentage of violations

#     while niterations < max_iterations && nincrements < 5
#         niterations += 1
#         old_C = C

#         # Calculate gradient descent and cost
#         C, ∇C = gradient(loss, triplets, X)

#         # Debias the loss and gradient
#         @info "Debiased"
#         C = ((1 - loss.ρ) * C⁺ - loss.ρ * C⁻) / (1 - 2loss.ρ)
#         ∇C = ((1 - loss.ρ) .* ∇C⁺ .- loss.ρ .* ∇C⁻) ./ (1 - 2loss.ρ)

#         # Update the embedding according to the gradient
#         X.X = X.X .- (η / length(triplets) * nitems(X)) * ∇C
#         # X.X = X.X/norm(X.X)

#         if C < best_C
#             best_C = C
#             best_X = X
#         end

#         # Update learning rate
#         if old_C > C + tolerance
#             nincrements = 0
#             η *= 1.01
#         else
#             nincrements += 1
#             η *= 0.5
#         end

#         # Print out progress
#         if verbose && (niterations % print_every == 0)
#             # If we have a large number of triplets, computing the number of violations
#             # can be costly. Therefore, we only perform this operation every 10 iterations
#             # If more details are needed, you can set the environment variable
#             # JULIA_DEBUG=TripletEmbeddings before starting Julia.
#             pviolations = percent_violations(triplets, X)
#             @info @sprintf "loss = %s, iteration = %d, cost = %.2f, misclassifications = %.2f%%, ||∇ₓR|| = %.2f" typeof(loss) niterations C/size(triplets)[1] 100 * pviolations norm(∇C)
#         end

#         @debug "Iteration = $niterations, Cost = $C, nincrements = $nincrements" X.X percent_violations(triplets, X)
#     end

#     if !verbose
#         pviolations = percent_violations(triplets, X)
#     end

#     X = best_X
#     return pviolations

# end

function percent_violations(triplets::LabeledTriplets, X::Embedding)
    D = pairwise(SqEuclidean(), X.X, dims=2)

    nviolations = 0

    for t in triplets
        nviolations += (D[t[:i], t[:j]] > D[t[:i], t[:k]] && t[:y] == -1) || ((D[t[:i], t[:j]] < D[t[:i], t[:k]] && t[:y] == +1))
    end

    return nviolations/length(triplets)
end