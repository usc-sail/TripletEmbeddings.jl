"""
    Triplets(X::AbstractMatrix{<:Real})

Generate a set of triplets from X (matrix or embedding).

X is assumed to be an d × n AbstractMatrix, so it can be:
  - A d × n AbstractMatrix
  - A d × n TripletEmbeddings.Embedding
"""
struct Triplets{T} <: AbstractArray{T,1}
    triplets::Vector{T}

    function Triplets(X::AbstractMatrix{T}) where T
        triplets = label(X)
        @assert checktriplets(triplets) "Triplets do not contain all items"
        new{Tuple{Int,Int,Int}}(triplets)
    end

    function Triplets(X::AbstractVector{T}) where T
        @assert length(x) > 1 "Number of elements in X must be > 1"
        Triplets(reshape(X, 1, length(X)))
    end

    function Triplets(triplets::Vector{Tuple{Int,Int,Int}})
        @assert checktriplets(triplets) "Triplets do not contain all items"
        new{Tuple{Int,Int,Int}}(triplets)
    end

end

function checktriplets(triplets::Matrix{Int})
    return sort(unique(triplets)) == 1:maximum(triplets)
end

function checktriplets(triplets::Array{Tuple{Int64,Int64,Int64},1})
    return checktriplets(getindex.(triplets, [1 2 3]))
end

Base.size(triplets::Triplets) = size(triplets.triplets)
Base.getindex(triplets::Triplets, inds...) = getindex(triplets.triplets, inds...)

ntriplets(triplets::Triplets) = length(triplets)

"""
    label(X::Matrix{T}) where T

Compute the triplets from X. X is assumed to be d × n,
so that each item is a vector.
"""
function label(X::Matrix{T}) where T <: Real
    d, n = size(X)
    D = pairwise(SqEuclidean(), X, dims=2)
    
    triplets = Vector{Tuple{Int,Int,Int}}(undef, n*binomial(n-1, 2))
    counter = 0

    for k = 1:n, j = 1:k-1, i = 1:n
        if i != j && i != k
            if D[i,j] < D[i,k]
                counter +=1
                @inbounds triplets[counter] = (i, j, k)
            elseif D[i,j] > D[i,k]
                counter += 1
                @inbounds triplets[counter] = (i, k, j)
            end
        end
    end

    return triplets[1:counter]
end

function label(X::AbstractEmbedding{T}) where T <: Real
    return label(X.X)
end

# probability represents the probability of swapping the order of a
# random triplet
function label(X::Matrix{T}, p::Array{S,3}) where {T <: Real, S <: Real}
    d, n = size(X)
    D = pairwise(SqEuclidean(), X, dims=2)

    triplets = Vector{Tuple{Int,Int,Int}}(undef, n*binomial(n-1, 2))
    counter = 0    

    for k = 1:n, j = 1:k-1, i = 1:n
        if i != j && i != k

            @inbounds mistake = p[i,j,k] .<= 1 - rand()

            if D[i,j] < D[i,k]
                counter +=1
                if !mistake
                    @inbounds triplets[counter,:] = [i, j, k]
                else
                    @inbounds triplets[counter,:] = [i, k, j]
                end
            elseif D[i,j] > D[i,k]
                counter += 1
                if !mistake
                    @inbounds triplets[counter,:] = [i, k, j]
                else
                    @inbounds triplets[counter,:] = [i, j, k]
                end
            end
        end
    end

    return triplets[1:counter,:]
end