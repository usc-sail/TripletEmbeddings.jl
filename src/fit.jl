function fit!(
    loss::AbstractLoss,
    triplets::Triplets{<:Integer},
    X::Embedding;
    η::Real=1.0,
    verbose::Bool = true,
    print_every::Int = 10,
    max_iterations::Int64 = 1000
)

    @assert max_iterations >= 10 "Iterations must be at least 10"
    @assert 1 ≤ print_every ≤ max_iterations || throw(ArgumentError("print_every should be in the interval [1, max_iterations]"))

    # maximum(getindex.(triplets, [1 2 3])) == nitems(X) || throw(ArgumentError("Number of items to embed must equal to the number of items represented by triplet comparisons"))

    C = Inf                      # Cost
    ∇C = zeros(Float64, size(X)) # Gradient

    tolerance = 1e-7 # convergence tolerance
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
    D = pairwise(SqEuclidean(), X, dims=2)

    nviolations = 0

    for t = 1:ntriplets(triplets)
        nviolations += D[triplets[t][1], triplets[t][2]] > D[triplets[t][1], triplets[t][3]]
    end

    return nviolations/ntriplets(triplets)
end