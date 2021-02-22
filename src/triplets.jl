"""
    Triplets(X::AbstractMatrix{<:Real}[, f::Function = x -> 1])

Generate a set of triplets from X (matrix or embedding).

X is assumed to be an d × n AbstractMatrix, so it can be:
  - A d × n AbstractMatrix
  - A d × n TripletEmbeddings.Embedding
"""
struct Triplets{T} <: AbstractArray{T,1}
    triplets::Vector{T}

    function Triplets(X::AbstractMatrix{S}; f::Function = x -> 1, shuffle::Bool = false) where S <: Real
        triplets = label(X, f, shuffle)
        checktriplets(triplets) || throw(AssertionError("Triplets do not contain all items."))
        new{Tuple{Int32,Int32,Int32}}(triplets)
    end

    function Triplets(X::AbstractVector{S}; f::Function = x -> 1) where S <: Real
        @assert length(X) > 1 "Number of elements in X must be > 1"
        Triplets(reshape(X, 1, length(X)))
    end
end

"""
    checktriplets(triplets::Vector{Vector{Int32,Int32,Int32}})

Checks whether triplets contains all elements from 1:n, where n is the maximum
value found in triplets.
"""
function checktriplets(triplets::Vector{Tuple{Int32,Int32,Int32}})
    # Transform triplets into Matrix{Int32} and call respective function
    return checktriplets(getindex.(triplets, [1 2 3]))
end

function checktriplets(triplets::Matrix{Int32})
    return sort(unique(triplets)) == 1:maximum(triplets)
end

# createinstance(::Type{<:Triplets{T}}, args...) where {T} = Triplets{T}(args)

Base.size(triplets::Triplets) = size(triplets.triplets)
# Base.getindex(triplets::Triplets, inds...) = Triplets(getindex(triplets.triplets, inds...))
Base.getindex(triplets::Triplets, inds...) = getindex(triplets.triplets, inds...)



ntriplets(triplets::Triplets) = length(triplets)

@doc raw"""
    label(X::Matrix{T}, f::Function) where T <: Real

Compute the triplets from X. X is assumed to be d × n,
so that each item of the embedding is a vector (column).

f is a function modeling the probabilities of success/failure,
assuming that these probabilities are modeled through a Bernoulli
random variable:

```math
y_{ijk} = \cases{
    -1 \quad \text{w.p.} & f(D_{ij}^* - D_{ik}^*) \\
    +1 \quad \text{w.p.} & 1- f(D_{ij}^* - D_{ik}^*)
}
```
"""
function label(X::AbstractMatrix{T}, f::Function, shuffle::Bool) where T <: Real
    maximum(X) ≠ minimum(X) || throw(ArgumentError("Embedding is constant, no triplets to compute."))

    d, n = size(X)
    D = pairwise(SqEuclidean(), X, dims=2)

    triplets = Vector{Tuple{Int32,Int32,Int32}}(undef, n*binomial(n-1, 2))
    counter = 0

    for k = 1:n, j = 1:k-1, i = 1:n
        if i != j && i != k

            # Random noise distributes Bernoulli, with p = f(abs(D[i,j] - D[i,k]))
            # If f(x) = p (constant), then the noise is independent of the distance between pairs
            @inbounds mistake = f(abs(D[i,j] - D[i,k])) <= 1 - rand()

            if D[i,j] < D[i,k]
                counter +=1
                if !mistake
                    @inbounds triplets[counter] = (i, j, k)
                else
                    @inbounds triplets[counter] = (i, k, j)
                end
            elseif D[i,j] > D[i,k]
                counter += 1
                if !mistake
                    @inbounds triplets[counter] = (i, k, j)
                else
                    @inbounds triplets[counter] = (i, j, k)
                end
            end
        end
    end

    return shuffle ? Random.shuffle(triplets[1:counter]) : triplets[1:counter]

end