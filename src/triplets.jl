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
Triplet(t::NTuple{3,T}) where T <: Integer = Triplet{T}((i = t[1], j = t[2], k = t[3]))

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
    Triplet{T}((i,j,k))
end

function Triplet(S::Type{U}, i::T, j::T, k::T) where {U <: Integer, T <: Integer}
    Triplet{S}((i,j,k))
end

function Base.show(io::IO, ::Type{Triplet})
    print(io, "Triplet")
end

function Base.show(io::IO, t::Triplet)
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
    print(io, "Triplet{$(eltype(t))}(")
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
    Triplets(X::AbstractMatrix{T}; f::Function = x -> 1, shuffle::Bool = false, percentage::Real = 100) where T <: Real

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
function Triplets(X::AbstractMatrix{T}; f::Function = x -> 1, shuffle::Bool = false, percentage::Real = 100) where T <: Real
    0 < percentage ≤ 100 || throw(ArgumentError("Percentage of triplets must be in the interval (0, 100]"))
    # if percentage < 100
    #     @warn "Shuffling to have all items represented"
    #     shuffle = true
    # end
    triplets = label(X, f, shuffle)[1:floor(Int, end * percentage / 100)]
    checktriplets(triplets) || throw(AssertionError("Triplets do not contain all items."))
    return triplets
end

function Triplets(S::Type{U}, X::AbstractMatrix{T}; f::Function = x -> 1, shuffle::Bool = false, percentage::Real = 100) where {U <: Integer, T <: Real}
    0 < percentage ≤ 100 || throw(ArgumentError("Percentage of triplets must be in the interval (0, 100]"))
    # if percentage < 100
    #     @warn "Shuffling to have all items represented"
    #     shuffle = true
    # end
    triplets = label(X, f, shuffle, S)[1:floor(Int, end * percentage / 100)]
    checktriplets(triplets) || throw(AssertionError("Triplets do not contain all items."))
    return triplets
end

function Triplets(X::AbstractVector{S}; f::Function = x -> 1, shuffle::Bool = false) where S <: Real
    length(X) > 1 || throw(AssertionError("Number of elements in X must be > 1"))
    Triplets(reshape(X, 1, length(X)); f = f, shuffle = shuffle)
end

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
    sum(isnan.(X)) == 0 || throw(DomainError("Embedding X has NaNs values."))

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
    label(X, f, shuffle, tripletstype(n))
end

function tripletstype(n::Int)
    S = if typemax(Int16) > n
        Int16
    elseif typemax(UInt16) > n
        UInt16
    elseif typemax(Int32) > n
        Int32
    elseif typemax(UInt32) > n
        UInt32
    else
        Int64
    end
    return S
end

"""
    sampletriplet([rng], n)

Draw a triplet of distinct integers between 1 and `n` without replacement.

Optionally specify a random number generator `rng` as the first argument
(defaults to `Random.GLOBAL_RNG`).

This function return a NTuple{3,Int}, _not_ a Triplet{T}.

"""
function sampletriplet(rng::AbstractRNG, n::Int)
    i = rand(rng, 1:n)
    j = rand(rng, 1:n-1)
    k = rand(rng, 1:n)

    return (i, i == j ? n : j, (k == i || k == j) ? rand(rng, (1:n)[Not(i,j)]) : k)
end

sampletriplet(n::Int) = sampletriplet(Random.GLOBAL_RNG, n)


"""
    sampletriplets([rng], n, number_of_triplets)

Draw `number_of_triplets` triplets of distinct integers between 1 and `n` without replacement.

Optionally specify a random number generator `rng` as the first argument
(defaults to `Random.GLOBAL_RNG`).

This function returns a Vector{NTuple{3,Int}}, _not_ a Vector{Triplet{T}}.

"""
function sampletriplets(rng::AbstractRNG, n::Int, number_of_triplets::Int)
    S = tripletstype(number_of_triplets)
    triplets = [sampletriplet(n) for _ in 1:number_of_triplets]
    return triplets
end

sampletriplets(n::Int, number_of_triplets::Int) = sampletriplets(Random.GLOBAL_RNG, n, number_of_triplets)



########################
### Labeled Triplets ###
########################

const LabeledTriplet{T} = NamedTuple{(:i, :j, :k, :y), Tuple{T, T, T, T}} where T <: Integer

@doc raw"""
    function LabeledTriplet([S::Type{U},] t::NTuple{4,T}) where {U <: Integer, T <: Integer}

Create a Labeled Triplet.

A Labeled Triplet always has the names (or keys) (:i, :j, :k. :y). For an embedding X with pairwise distances D[i,j],
y ∈ {-1,+1} denotes whether D[i,j] < D[i,k] (y = -1) or D[i,j] > D[i,k] (y = +1), according to:

    ```math
    \begin{equation}
        y_t =
        \begin{cases}
          -1, &\text{w.p. }\quad f(D^*_{ij} - D^*_{ik})\\
          +1, &\text{w.p. }\quad 1 - f(D^*_{ij} - D^*_{ik}).
        \end{cases}
    \end{equation}
    ```

The underlying struct for a Triplet is a NamedTuple with names = (:i, :j, :k).

---
Other constructors:
 - function LabeledTriplet(S::Type{U}, t::NTuple{4,T}) where {U <: Integer, T <: Integer}
 - function LabeledTriplet(t::Vector{T}) where T <: Integer
 - function LabeledTriplet(S::DataType, t::Vector{T}) where T <: Integer
 - function LabeledTriplet(i::Int, j::Int, k::Int, y::Int)
 - function LabeledTriplet(S::DataType, i::Int, j::Int, k::Int, y::Int)
"""
LabeledTriplet(t::NTuple{4,T}) where T <: Integer = LabeledTriplet{eltype(t)}((i = t[1], j = t[2], k = t[3], y = t[4]))

function LabeledTriplet(S::Type{U}, t::NTuple{4,T}) where {U <: Integer, T <: Integer}
    S <: Integer || throw(ArgumentError("S must be a subtype of Integer"))
    LabeledTriplet{S}(NTuple{4,S}(t))
end

function LabeledTriplet(t::Vector{T}) where T <: Integer
    length(t) == 4 || throw(ArgumentError("LabeledTriplet must be of length 4"))
    @assert t[end] in [-1,1] || throw(ArgumentError("Label y (last value of argument) must be either -1 or +1."))
    LabeledTriplet{T}(NTuple{4,T}(t))
end

function LabeledTriplet(S::Type{U}, t::Vector{T}) where {U <: Integer, T <: Integer}
    length(t) == 4 || throw(ArgumentError("LabeledTriplet must be of length 4"))
    @assert t[end] in [-1,1] || throw(ArgumentError("Label y (last value of argument) must be either -1 or +1."))
    LabeledTriplet{S}(NTuple{4,S}(t))
end

function LabeledTriplet(i::T, j::T, k::T, y::T) where T <: Integer
    @assert y in [-1,1] || throw(ArgumentError("Label y must be either -1 or +1."))
    LabeledTriplet{eltype(i)}((i,j,k,y))
end

function LabeledTriplet(S::Type{U}, i::T, j::T, k::T, y::T) where {U <: Integer, T <: Integer}
    @assert y in [-1,1] || throw(ArgumentError("Label y must be either -1 or +1."))
    LabeledTriplet{S}((i,j,k,y))
end

function Base.show(io::IO, ::Type{LabeledTriplet})
    print(io, "LabeledTriplet")
end

function Base.show(io::IO, t::LabeledTriplet)
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
    print(io, "LabeledTriplet{$(eltype(t))}(")
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


const LabeledTriplets{T} = Vector{LabeledTriplet{T}} where T <: Integer

function Base.show(io::IO, ::Type{LabeledTriplets{T}}) where T
    print(io, "LabeledTriplets{$(T)}")
end

"""
    LabeledTriplets(X::AbstractMatrix{T}; f::Function = x -> 1, shuffle::Bool = false, percentage::Real = 100) where T <: Real

Generate a set of labeled triplets from X (matrix or embedding).

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
function LabeledTriplets(X::AbstractMatrix{T}; f::Function = x -> 1, shuffle::Bool = false, percentage::Real = 100) where T <: Real
    0 < percentage ≤ 100 || throw(ArgumentError("Percentage of triplets must be in the interval (0, 100]"))
    # if percentage < 100
    #     @warn "Shuffling to have all items represented"
    #     shuffle = true
    # end
    triplets = label(LabeledTriplet, X, f, shuffle)[1:floor(Int, end * percentage / 100)]
    checktriplets(triplets) || throw(AssertionError("Triplets do not contain all items."))
    return triplets
end

function LabeledTriplets(S::Type{U}, X::AbstractMatrix{T}; f::Function = x -> 1, shuffle::Bool = false, percentage::Real = 100) where {U <: Integer, T <: Real}
    0 < percentage ≤ 100 || throw(ArgumentError("Percentage of triplets must be in the interval (0, 100]"))
    # if percentage < 100
    #     @warn "Shuffling to have all items represented"
    #     shuffle = true
    # end
    triplets = label(LabeledTriplet, X, f, shuffle, S)[1:floor(Int, end * percentage / 100)]
    checktriplets(triplets) || throw(AssertionError("Triplets do not contain all items."))
    return triplets
end

function LabeledTriplets(X::AbstractVector{S}; f::Function = x -> 1, shuffle::Bool = false) where S <: Real
    length(X) > 1 || throw(AssertionError("Number of elements in X must be > 1"))
    LabeledTriplets(reshape(X, 1, length(X)); f = f, shuffle = shuffle)
end

"""
    checktriplets(triplets::LabeledTriplets)

Checks whether triplets contains all elements from 1:n, where n is the maximum
value found in triplets.
"""
function checktriplets(triplets::LabeledTriplets{T}) where T <: Integer
    # Transform triplets into Matrix{Int32} and call respective function
    return checktriplets(getindex.(triplets, [1 2 3]))
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
function label(::Type{LabeledTriplet}, X::AbstractMatrix{T}, f::Function, shuffle::Bool, U::Type{V}) where {T <: Real, V <: Integer}
    maximum(X) ≠ minimum(X) || throw(DomainError("Embedding is constant, no triplets to compute."))
    sum(isnan.(X)) == 0 || throw(DomainError("Embedding X has NaNs values."))

    d, n = size(X)
    D = pairwise(SqEuclidean(), X, dims=2)

    triplets = Vector{LabeledTriplet{U}}(undef, n*binomial(n-1, 2))
    counter = 0

    for k = 1:n, j = 1:k-1, i = 1:n
        if i != j && i != k

            # Random noise distributes Bernoulli, with p = f(abs(D[i,j] - D[i,k]))
            # If f(x) = p (constant), then the noise is independent of the distance between pairs
            @inbounds mistake = f(abs(D[i,j] - D[i,k])) <= 1 - rand()

            if D[i,j] < D[i,k]
                counter +=1
                @inbounds triplets[counter] = LabeledTriplet{U}((i, j, k, !mistake ? -1 : +1))
            elseif D[i,j] > D[i,k]
                counter += 1
                @inbounds triplets[counter] = LabeledTriplet{U}((i, j, k, !mistake ? +1 : -1))
            end
        end
    end

    return shuffle ? Random.shuffle(triplets[1:counter]) : triplets[1:counter]

end

function label(::Type{LabeledTriplet}, X::AbstractMatrix{T}, f::Function, shuffle::Bool) where T <: Real
    d, n = size(X)
    label(LabeledTriplet, X, f, shuffle, tripletstype(n))
end