"""
    Triplets(data::AbstractMatrix{<:Real})

Generate a set of triplets from data (or an embedding).

data is assumed to be an d × n AbstractMatrix, so it can be:
  - A d × n AbstractMatrix
  - A d × n TripletEmbeddings.Embedding
"""
struct Triplets{T} <: AbstractArray{T,1}
    triplets::Vector{T}

    function Triplets(data::AbstractMatrix{<:Real})
        d, n = size(data)
        D = pairwise(SqEuclidean(), data, dims=2)
        
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

        new{Tuple{Int,Int,Int}}(triplets[1:counter])
    end

    function Triplets(data::AbstractVector{<:Real})
        Triplets(reshape(data, 1, length(data)))
    end

    function Triplets(triplets::Vector{Tuple{Int,Int,Int}})
        new{Tuple{Int,Int,Int}}(triplets)
    end

end

Base.size(triplets::Triplets) = size(triplets.triplets)
Base.getindex(triplets::Triplets, inds...) = getindex(triplets.triplets, inds...)

ntriplets(triplets::Triplets) = length(triplets)