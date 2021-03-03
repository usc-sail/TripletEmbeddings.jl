const Triplet{T} = NamedTuple{(:i, :j, :k), Tuple{T, T, T}} where T <: Integer

"""
    function Triplet([S::Type{U},] t::NTuple{3,T}) where {U <: Integer, T <: Integer}

Create a Triplet.

A Triplet always has the names (or keys) (:i, :j, :k), and it is assumed that for an embedding X and pairwise distances D[i,j],
D[i,j] < D[i,k] (so i is assumed or annotated to be closer to j than k).

The underlying struct for a Triplet is a NamedTuple with names = (:i, :j, :k).

---
Other constructors:
 - function Triplet(S::Type{U}, t::NTuple{3,T}) where {U <: Integer, T <: Integer}
 - function Triplet(t::Vector{T}) where T <: Integer
 - function Triplet(S::DataType, t::Vector{T}) where T <: Integer
 - function Triplet(i::Int, j::Int, k::Int)
 - function Triplet(S::DataType, i::Int, j::Int, k::Int)
"""
Triplet(t::NTuple{3,T}) where T <: Integer = Triplet((i = t[1], j = t[2], k = t[3]))

function Triplet(S::Type{U}, t::NTuple{3,T}) where {U <: Integer, T <: Integer}
    S <: Integer || throw(ArgumentError("S must be a subtype of Integer"))
    Triplet{S}(NTuple{3,S}(t))
end

function Triplet(t::Vector{T}) where T <: Integer
    length(t) == 3 || throw(ArgumentError("Triplet must be of length 3"))
    Triplet{T}(NTuple{3,T}(t))
end

function Triplet(S::Type{U}, t::Vector{T}) where {U <: Integer, T <: Integer}
    length(t) == 3 || throw(ArgumentError("Triplet must be of length 3"))
    Triplet{S}(NTuple{3,S}(t))
end

function Triplet(i::T, j::T, k::T) where T <: Integer
    Triplet{eltype(i)}((i,j,k))
end

function Triplet(S::Type{U}, i::T, j::T, k::T) where {U <: Integer, T <: Integer}
    Triplet{S}((i,j,k))
end

Triplet() = Triplet(0, 0, 0)

Base.show(io::IO, ::Type{Triplet}) = print(io, "Triplet{$(T)}")

function Base.show(io::IO, t::Triplet{T}) where T <: Integer
    n = nfields(t)
    for i = 1:n
        # if field types aren't concrete, show full type
        if typeof(getfield(t, i)) !== fieldtype(typeof(t), i)
            show(io, typeof(t))
            print(io, "(")
            show(io, Tuple(t))
            print(io, ")")
            return
        end
    end

    typeinfo = get(io, :typeinfo, Any)
    print(io, "Triplet{$T}(")
    for i = 1:n
        print(io, fieldname(typeof(t),i), " = ")
        show(IOContext(io, :typeinfo =>
                       t isa typeinfo <: NamedTuple ? fieldtype(typeinfo, i) : Any),
             getfield(t, i))
        if n == 1
            print(io, ",")
        elseif i < n
            print(io, ", ")
        end
    end
    print(io, ")")
end


const Triplets{T} = Vector{Triplet{T}} where T <: Integer

function Base.show(io::IO, ::Type{Triplets{T}}) where T
    print(io, "Triplets{$(T)}")
end

"""
    Triplets(X::AbstractMatrix{T}[, f::Function = x -> 1, shuffle::Bool = false]) where T <: Real

Generate a set of triplets from X (matrix or embedding).

X is assumed to be an d × n AbstractMatrix, so it can be:
  - A d × n AbstractMatrix
  - A d × n TripletEmbeddings.Embedding

Parameters:
 - f::Function corresponds to the noise function when generating triplets (x -> 1 implies probability of success is 1)
 - shuffle::Bool is a flag deciding whether the triplets are returned shuffled or not (default is false)

This method will optimize the type used for each Triplet (between Int16 and Int64).
---
    Triplets(triplets::Vector{Triplet{(:i, :j, :k), Tuple{T, T, T}}}) where T <: Integer

Function to generate Triplets from a Vector{Triplet}. To be used only with Stochastic Gradient Descent (SGD).

"""
function Triplets(X::AbstractMatrix{T}; f::Function = x -> 1, shuffle::Bool = false, percentage::Int = 100) where T <: Real
    0 < percentage ≤ 100 || throw(ArgumentError("Percentage of triplets must be in the interval (0, 100]"))
    triplets = label(X, f, shuffle)[1:floor(Int, end * percentage / 100)]
    checktriplets(triplets) || throw(AssertionError("Triplets do not contain all items."))
    return triplets
end

function Triplets(S::Type{U}, X::AbstractMatrix{T}; f::Function = x -> 1, shuffle::Bool = false, percentage::Int = 100) where {U <: Integer, T <: Real}
    0 < percentage ≤ 100 || throw(ArgumentError("Percentage of triplets must be in the interval (0, 100]"))
    triplets = label(X, f, shuffle, S)[1:floor(Int, end * percentage / 100)]
    checktriplets(triplets) || throw(AssertionError("Triplets do not contain all items."))
    return triplets
end

function Triplets(X::AbstractVector{S}; f::Function = x -> 1, shuffle::Bool = false) where S <: Real
    length(X) > 1 || throw(AssertionError("Number of elements in X must be > 1"))
    Triplets(reshape(X, 1, length(X)); f = f, shuffle = shuffle)
end

Triplets() = Triplets{Int8}(undef, 0)

"""
    checktriplets(triplets::Triplets)

Checks whether triplets contains all elements from 1:n, where n is the maximum
value found in triplets.
"""
function checktriplets(triplets::Triplets{T}) where T <: Integer
    # Transform triplets into Matrix{Int32} and call respective function
    return checktriplets(getindex.(triplets, [1 2 3]))
end

function checktriplets(triplets::Matrix{T}) where T <: Integer
    return sort(unique(triplets)) == 1:maximum(triplets)
end

function ntriplets(triplets::Triplets{T}) where T <: Integer
    length(triplets)
end

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
function label(X::AbstractMatrix{T}, f::Function, shuffle::Bool, S::Type{U}) where {T <: Real, U <: Integer}
    maximum(X) ≠ minimum(X) || throw(ArgumentError("Embedding is constant, no triplets to compute."))

    d, n = size(X)
    D = pairwise(SqEuclidean(), X, dims=2)

    triplets = Vector{Triplet{S}}(undef, n*binomial(n-1, 2))
    counter = 0

    for k = 1:n, j = 1:k-1, i = 1:n
        if i != j && i != k

            # Random noise distributes Bernoulli, with p = f(abs(D[i,j] - D[i,k]))
            # If f(x) = p (constant), then the noise is independent of the distance between pairs
            @inbounds mistake = f(abs(D[i,j] - D[i,k])) <= 1 - rand()

            if D[i,j] < D[i,k]
                counter +=1
                if !mistake
                    @inbounds triplets[counter] = Triplet(S, i, j, k)
                else
                    @inbounds triplets[counter] = Triplet(S, i, k, j)
                end
            elseif D[i,j] > D[i,k]
                counter += 1
                if !mistake
                    @inbounds triplets[counter] = Triplet(S, i, k, j)
                else
                    @inbounds triplets[counter] = Triplet(S, i, j, k)
                end
            end
        end
    end

    return shuffle ? Random.shuffle(triplets[1:counter]) : triplets[1:counter]

end

function label(X::AbstractMatrix{T}, f::Function, shuffle::Bool) where T <: Real
    d, n = size(X)
    S = if typemax(Int16) > (n * binomial(n-1, 2))
        Int16
    elseif typemax(UInt16) > (n * binomial(n-1, 2))
        UInt16
    elseif typemax(Int32) > (n * binomial(n-1, 2))
        Int32
    elseif typemax(UInt32) > (n * binomial(n-1, 2))
        UInt32
    else
        Int64
    end
    label(X, f, shuffle, S)
end