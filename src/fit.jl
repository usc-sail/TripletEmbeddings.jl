function fit!(
    loss::AbstractLoss,
    triplets::Triplets,
    X::AbstractEmbedding;
    verbose::Bool = true,
    max_iterations::Int64 = 1000,
    debug::Bool = false
)

    @assert max_iterations >= 10 "Iterations must be at least 10"
    @assert maximum([maximum(t) for t in triplets]) == nitems(X)
    
    if verbose
        println("Fitting embedding with loss $(typeof(loss))")
    end

    if debug
        iteration_Xs = zeros(Float64, ndims(X), nitems(X), max_iterations)
    end   

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
        X = X - (η / ntriplets(triplets) * nitems(X)) * ∇C

        if C < best_C
            best_C = C
            best_X = X
        end

        # Save each iteration if indicated
        if debug
            iteration_Xs[:,:,niterations] = X.X
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
            pviolations = percent_violations(triplets, X)
            @printf("iteration # = %d, cost = %.2f, violations = %.2f %%\n", niterations, C, 100 * pviolations)
        end
    end

    if !verbose
        pviolations = percent_violations(triplets, X)
    end

    if debug
        return iteration_Xs[:,:,1:no_iterations], pviolations
    else
        X = best_X
        return pviolations
    end
end

function percent_violations(triplets::Triplets, X::AbstractEmbedding)
    D = pairwise(SqEuclidean(), X, dims=2)

    nviolations = 0

    for t = 1:ntriplets(triplets)
        nviolations += D[triplets[t][1], triplets[t][2]] > D[triplets[t][1], triplets[t][3]]
    end

    return nviolations/ntriplets(triplets)
end