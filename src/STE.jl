struct STE <: AbstractLoss
    σ::Real
    constant::Float64

    function STE(;σ::T = 1/sqrt(2)) where T <: Real
        if σ ≤ 0
            throw(ArgumentError("σ in STE loss must be > 0"))
        end
        new(σ, 1/σ^2)
    end

end

function kernel(loss::STE, X::Embedding)
    K = pairwise(SqEuclidean(), X, dims=2)
    c = -loss.constant / 2

    for j in 1:nitems(X), i in 1:nitems(X)
        @inbounds K[i,j] = exp(c * K[i,j])
    end
    return K
end

function gradient(loss::STE, triplets::Triplets, X::Embedding)

    K = kernel(loss, X) # Triplet kernel values (in the STE loss)

    # We need to create an array to prevent race conditions
    # This is the best average solution for small and big Embeddings
    nthreads = Threads.nthreads()
    triplets_range = partition(ntriplets(triplets), nthreads)

    C = zeros(Float64, nthreads)
    ∇C = [zeros(Float64, size(X)) for _ in 1:nthreads]

    Threads.@threads for tid in 1:nthreads
        C[tid] = tgradient!(loss, triplets, X, K, ∇C[tid], triplets_range[tid])
    end

    return sum(C), -sum(∇C)
end

function tgradient!(
    loss::STE,
    triplets::Triplets,
    X::Embedding,
    K::Matrix{<:AbstractFloat},
    ∇C::Matrix{<:AbstractFloat},
    triplets_range::UnitRange{Int64})

    C = 0.0

    for t in triplets_range
        @views @inbounds i, j, k = triplets[t][:i], triplets[t][:j], triplets[t][:k]

        @inbounds P = K[i,j] / (K[i,j] + K[i,k])
        C += -log(P)

        for d in 1:ndims(X)
            @inbounds dx_j = (1 - P) * (X[d,i] - X[d,j])
            @inbounds dx_k = (1 - P) * (X[d,i] - X[d,k])

            @inbounds ∇C[d,i] += - loss.constant * (dx_j - dx_k)
            @inbounds ∇C[d,j] +=   loss.constant *  dx_j
            @inbounds ∇C[d,k] += - loss.constant *  dx_k
        end
    end
    return C
end