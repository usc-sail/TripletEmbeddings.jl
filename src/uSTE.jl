"""
   uSTE <: AbstractLoss

_U_nbiased Stochastic Triplet Embedding loss.

This loss implements the paper "Learning with Noisy Labels" for STE. The paper
may be found in https://proceedings.neurips.cc/paper/2013/file/3871bd64012152bfb53fdf04b401193f-Paper.pdf
"""
struct uSTE <: AbstractLoss
    σ::Real
    constant::Float64
    ρ::Dict{Int,Float64}

    function uSTE(;σ::S = 1/sqrt(2), ρ::Dict{Int,Float64} = Dict(-1 => 0.4, +1 => 0.4)) where {S <: Real, T <: AbstractFloat}
        σ > 0 || throw(ArgumentError("σ in STE loss must be > 0"))
        (0 ≤ ρ[-1] < 1 && 0 ≤ ρ[+1] < 1) || throw(ArgumentError("ρ₊ in uSTE loss must be in the (0, 1) interval."))
        new(σ, 1/σ^2, ρ)
    end
end

function Base.show(io::IO, loss::uSTE)
    println("uSTE(σ = $(round(loss.σ, digits=3)), constant = $(round(loss.constant, digits=3)), ρ₊ = $(round(loss.ρ[1], digits=3)), ρ₋ = $(round(loss.ρ[-1], digits=3)))")
end

@doc raw"""
    function kernel(loss::uSTE, X::AbstractMatrix)

Computes:

```math
K = \exp(\|X_i - X_j\|^2/(2\sigma^2)) \forall (i,j) \in {1,\ldots,n} \times {1,\ldots,n}.
```
"""
function kernel(loss::uSTE, X::Embedding)
    K = pairwise(SqEuclidean(), X.X, dims=2)
    c = -loss.constant / 2

    for ij in eachindex(K)
        @inbounds K[ij] = exp(c * K[ij])
    end
    return K
end

function gradient(loss::uSTE, triplets::LabeledTriplets, X::AbstractMatrix)

    K = kernel(loss, X) # Triplet kernel values (in the STE loss)

    # We need to create an array to prevent race conditions
    # This is the best average solution for small and big Embeddings
    nthreads = Threads.nthreads()
    triplets_range = partition(length(triplets), nthreads)

    C⁺ = zeros(Float64, nthreads)
    ∇C⁺ = [zeros(Float64, size(X)) for _ in 1:nthreads]

    Threads.@threads for tid in 1:nthreads
        C⁺[tid] = tgradient!(∇C⁺[tid], loss, triplets, X, K, triplets_range[tid]; flip_triplets=false)
    end

    C⁻ = zeros(Float64, nthreads)
    ∇C⁻ = [zeros(Float64, size(X)) for _ in 1:nthreads]

    Threads.@threads for tid in 1:nthreads
        C⁻[tid] = tgradient!(∇C⁻[tid], loss, triplets, X, K, triplets_range[tid]; flip_triplets=true)
    end

    return (sum(C⁺) + sum(C⁻)) / (1 - sum(values(loss.ρ))), -(sum(∇C⁺) + sum(∇C⁻)) / (1 - sum(values(loss.ρ)))
end

function tgradient!(
    ∇C::Matrix{<:AbstractFloat},
    loss::uSTE,
    triplets::LabeledTriplets,
    X::AbstractMatrix,
    K::Matrix{<:AbstractFloat},
    triplets_range::UnitRange{Int64};
    flip_triplets::Bool = false,
    )

    C = 0.0

    for t in triplets_range
        @inbounds i, j, k = if triplets[t][:y] == -1
            triplets[t][:i], triplets[t][:j], triplets[t][:k]
        else
            triplets[t][:i], triplets[t][:k], triplets[t][:j]
        end

        if flip_triplets
            i, j, k = i, k, j
        end

        @inbounds P = K[i,j] / (K[i,j] + K[i,k])
        C += if flip_triplets
            -loss.ρ[triplets[t][:y]] * log(P)
        else
            -(1 - loss.ρ[-triplets[t][:y]]) * log(P)
        end

        for d in 1:ndims(X)
            @inbounds ∂x_j = (1 - P) * (X[d,i] - X[d,j])
            @inbounds ∂x_k = (1 - P) * (X[d,i] - X[d,k])

            if flip_triplets
                @inbounds ∇C[d,i] += - loss.constant * (∂x_j - ∂x_k) * loss.ρ[triplets[t][:y]]
                @inbounds ∇C[d,j] +=   loss.constant *  ∂x_j         * loss.ρ[triplets[t][:y]]
                @inbounds ∇C[d,k] += - loss.constant *  ∂x_k         * loss.ρ[triplets[t][:y]]
            else
                @inbounds ∇C[d,i] += - loss.constant * (∂x_j - ∂x_k) * (1 - loss.ρ[-triplets[t][:y]])
                @inbounds ∇C[d,j] +=   loss.constant *  ∂x_j         * (1 - loss.ρ[-triplets[t][:y]])
                @inbounds ∇C[d,k] += - loss.constant *  ∂x_k         * (1 - loss.ρ[-triplets[t][:y]])
            end
        end
    end
    return C
end