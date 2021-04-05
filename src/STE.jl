struct STE <: AbstractLoss
    σ::Real
    constant::Float64

    function STE(;σ::T = 1/sqrt(2)) where T <: Real
        σ > 0 || throw(ArgumentError("σ in STE loss must be > 0"))
        new(σ, 1/σ^2)
    end

end

@doc raw"""
    function kernel(loss::STE, X::AbstractMatrix)

Computes:

```math
K = \exp(\|X_i - X_j\|^2/(2\sigma^2)) \forall (i,j) \in {1,\ldots,n} \times {1,\ldots,n}.
```
"""
function kernel(loss::STE, X::Embedding)
    K = pairwise(SqEuclidean(), X.X, dims=2)
    c = -loss.constant / 2

    for ij in eachindex(K)
        @inbounds K[ij] = exp(c * K[ij])
    end
    return K
end

function gradient(loss::STE, triplets, X::AbstractMatrix)

    K = kernel(loss, X) # Triplet kernel values (in the STE loss)

    # We need to create an array to prevent race conditions
    # This is the best average solution for small and big Embeddings
    nthreads = Threads.nthreads()
    triplets_range = partition(length(triplets), nthreads)

    C = zeros(Float64, nthreads)
    ∇C = [zeros(Float64, size(X)) for _ in 1:nthreads]

    Threads.@threads for tid in 1:nthreads
        C[tid] = tgradient!(∇C[tid], loss, triplets, X, K, triplets_range[tid])
    end

    return sum(C), -sum(∇C)
end

function tgradient!(
    ∇C::Matrix{<:AbstractFloat},
    loss::STE,
    triplets::Triplets,
    X::AbstractMatrix,
    K::Matrix{<:AbstractFloat},
    triplets_range::UnitRange{Int64})

    C = 0.0

    for t in triplets_range
        @views @inbounds i, j, k = triplets[t][:i], triplets[t][:j], triplets[t][:k]

        @inbounds P = K[i,j] / (K[i,j] + K[i,k])
        C += -log(P)

        for d in 1:ndims(X)
            @inbounds ∂x_j = (1 - P) * (X[d,i] - X[d,j])
            @inbounds ∂x_k = (1 - P) * (X[d,i] - X[d,k])

            @inbounds ∇C[d,i] += - loss.constant * (∂x_j - ∂x_k)
            @inbounds ∇C[d,j] +=   loss.constant *  ∂x_j
            @inbounds ∇C[d,k] += - loss.constant *  ∂x_k
        end
    end
    return C
end


function tgradient!(
    ∇C::Matrix{<:AbstractFloat},
    loss::STE,
    triplets::LabeledTriplets,
    X::AbstractMatrix,
    K::Matrix{<:AbstractFloat},
    triplets_range::UnitRange{Int64})

    C = 0.0

    for t in triplets_range
        @views @inbounds i, j, k = if triplets[t][:y] == -1
            triplets[t][:i], triplets[t][:j], triplets[t][:k]
        else
            triplets[t][:i], triplets[t][:k], triplets[t][:j]
        end

        @inbounds P = K[i,j] / (K[i,j] + K[i,k])
        C += -log(P)

        for d in 1:ndims(X)
            @inbounds ∂x_j = (1 - P) * (X[d,i] - X[d,j])
            @inbounds ∂x_k = (1 - P) * (X[d,i] - X[d,k])

            @inbounds ∇C[d,i] += - loss.constant * (∂x_j - ∂x_k)
            @inbounds ∇C[d,j] +=   loss.constant *  ∂x_j
            @inbounds ∇C[d,k] += - loss.constant *  ∂x_k
        end
    end
    return C
end