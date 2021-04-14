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

    R = fill(Inf, max_iterations) # risk
    ∇R = zeros(Float64, size(X))  # gradient

    tolerance = 1e-7 # convergence tolerance
    η = 1.0          # learning rate
    best_X = X       # best embedding found so far

    # Perform main iterations
    i = 1   # Number of iterations (so far)
    lr_increments = 0   # Number of learning rate increments
    pviolations = 0.0 # Percentage of violations

    while i < max_iterations && lr_increments < 5
        i += 1

        # Calculate gradient descent and risk
        R[i], ∇R = gradient(loss, triplets, X)
        if any(isnan.(∇R))
            break
        end

        # Update the embedding according to the gradient
        X.X = X.X .- (η / length(triplets) * nitems(X)) * ∇R

        if R[i-1] < minimum(R)
            best_X = X
        end

        # Update learning rate
        if R[i-1] > R[i] + tolerance
            lr_increments = 0
            η *= 1.01
        else
            lr_increments += 1
            η *= 0.5
        end

        # Print out progress
        if verbose && (i % print_every == 0)
            # If we have a large number of triplets, computing the number of violations
            # can be costly. Therefore, we only perform this operation every 10 iterations
            # If more details are needed, you can set the environment variable
            # JULIA_DEBUG=TripletEmbeddings before starting Julia.
            pviolations = percent_violations(triplets, X)
            @info @sprintf "loss = %s, iteration = %d, risk = %.6f, misclassifications = %.2f%%, ||∇ₓR|| = %.6f" typeof(loss) i R[i]/length(triplets) 100 * pviolations norm(∇R/length(triplets))
        end

        @debug "Iteration = $i, risk = $(R[i]), lr_increments = $lr_increments" X.X percent_violations(triplets, X)
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

function percent_violations(triplets::LabeledTriplets, X::Embedding)
    D = pairwise(SqEuclidean(), X.X, dims=2)

    nviolations = 0

    for t in triplets
        nviolations += (D[t[:i], t[:j]] > D[t[:i], t[:k]] && t[:y] == -1) || ((D[t[:i], t[:j]] < D[t[:i], t[:k]] && t[:y] == +1))
    end

    return nviolations/length(triplets)
end