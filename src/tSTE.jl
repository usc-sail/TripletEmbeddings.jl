struct tSTE <: AbstractLoss
    α::Int
    constant::Float64

    function tSTE(;α::Int = 3)
        if α < 1
            throw(ArumentError("α in tSTE loss must be ≥ 1"))
        end
        new(α, (α + 1) / α)
    end

end

function kernel(loss::tSTE, X::AbstractEmbedding)
    K = pairwise(SqEuclidean(), X, dims=2)
    Q = zeros(Float64, nitems(X), nitems(X))

    constant = ((loss.α + 1) / -2)

    for j in 1:nitems(X), i in 1:nitems(X)
        # We compute K[i,j] = (1 + ||x_i - x_j||_2^2/α) ^ (-(α+1)/2)
        base = (1 + K[i,j] / loss.α)
        @inbounds Q[i,j] = base ^ -1
        @inbounds K[i,j] = base ^ constant
    end
    return K, Q
end

# function tcost(loss::STE, triplet::Tuple{Int,Int,Int}, X::AbstractEmbedding)
#     @inbounds i = triplet[1]
#     @inbounds j = triplet[2]
#     @inbounds k = triplet[3]

#     @inbounds K_ij = exp( -loss.constant * norm(X[i,:] - X[j,:])^2 / 2)
#     @inbounds K_ik = exp( -loss.constant * norm(X[i,:] - X[k,:])^2 / 2)

#     @inbounds P = K_ij / (K_ij + K_ik)

#     return -log(P)
# end

function gradient(loss::tSTE, triplets::Triplets, X::AbstractEmbedding)

    K, Q = kernel(loss, X) # Triplet kernel values (in the tSTE loss)
    
    # We need to create an array to prevent race conditions
    # This is the best average solution for small and big Embeddings
    nthreads = Threads.nthreads()
    triplets_range = partition(ntriplets(triplets), nthreads)
    
    C = zeros(Float64, nthreads)
    ∇C = [zeros(Float64, size(X)) for _ in 1:nthreads]
    
    Threads.@threads for tid in 1:nthreads
        C[tid] = tgradient!(loss, triplets, X, K, Q, ∇C[tid], triplets_range[tid])
    end

    # If the Embedding is small, we can use multithreading over the triplets
    # ∇C = [zeros(Float64, nitems(X), ndims(X), ntriplets(triplets)) for _ = 1:nthreads]
    # Threads.@threads for t in 1:ntriplets(triplets)
    #     # ∇C[:,:,t] = tgradient(loss, triplets[t], X, K)
    # end
    
    # If the Embedding is big, we can use multiple processes.
    # This requires calling @everywhere using TripletEmbeddings
    # and adding processes through addprocs(),
    # and using Distributed inside TripletEmbeddings
    # ∇C = @distributed (+) for t in 1:ntriplets(triplets)
    #     tgradient(loss, triplets[t], X, K)
    # end
    # return -∇C

    return sum(C), -sum(∇C)
end

function tgradient!(
    loss::tSTE,
    triplets::Triplets,
    X::AbstractEmbedding,
    K::Matrix{<:AbstractFloat},
    Q::Matrix{<:AbstractFloat},
    ∇C::Matrix{<:AbstractFloat},
    triplets_range::UnitRange{Int64})

    C = 0.0
    
    for t in triplets_range
        @views @inbounds i, j, k = triplets[t]

        @inbounds P = K[i,j] / (K[i,j] + K[i,k])
        C += -log(P)

        for d in 1:ndims(X)
            @inbounds dx_j = (1 - P) * Q[i,j] * (X[d,i] - X[d,j])
            @inbounds dx_k = (1 - P) * Q[i,k] * (X[d,i] - X[d,k])

            @inbounds ∇C[d,i] +=   loss.constant * (dx_k - dx_j)
            @inbounds ∇C[d,j] +=   loss.constant *  dx_j
            @inbounds ∇C[d,k] += - loss.constant *  dx_k
        end
    end
    return C
end