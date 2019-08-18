abstract type AbstractEmbedding{T} <: AbstractMatrix{T} end

"""
# Summary
    Embedding{T} <: AbstractEmbedding{T}

# Fields
    X::Matrix{T} # Embedding  (each column is an item of the Embedding)
    G::Matrix{T} # Gram matrix associated to X

# Examples
Building an embedding from the desired dimensions:

    Embedding(d::Int, n::Int)

Construct a random d × n Embedding.

---
Building an embedding from a tuple:

    Embedding(t::Tuple{Int,Int})

Construct a random n × d Embedding from a tuple.
If typeof(tuple) == Tuple{Int64}, the embedding
is reshaped such that it is a matrix of 1 × n.

---
Building an embedding from a matrix:

    Embedding(X₀::Matrix{T}) where T <: Real

Construct an Embedding from a matrix of points.

We assume that the points are in a d × n matrix, where
  - d: dimensions
  - n: Number of points

If eltype(X₀) == Int, an AbstractFloat is forced.
"""
mutable struct Embedding{T} <: AbstractEmbedding{T}
    X::Matrix{T} # Embedding 
    G::Matrix{T} # Gram matrix associated to X

    function Embedding(d::Int, n::Int)
        @assert n ≥ d "n ≥ d is required"

        X = rand(d, n)
        
        new{eltype(X)}(X, X' * X)
    end

    function Embedding(t::Tuple{Int,Int})
        @assert t[2] ≥ t[1] "n ≥ d is required"

        X = rand(Float64, t)
        new{Float64}(X, X' * X)
    end

    function Embedding(t::Tuple{Int})
        X = rand(Float64, t)
        X = reshape(X, length(X), 1)
        new{Float64}(X, X' * X)
    end

    function Embedding(X₀::AbstractMatrix{T}) where T <: Real
        d, n = size(X₀)
        @assert n ≥ d "n ≥ d is required"

        # X = X₀ * V(d)
        X = X₀ .+ 0.0 # This forces X to be a Matrix{AbstractFloat}

        new{eltype(X)}(X, X' * X)
    end

    function Embedding(X₀::AbstractVector{T}) where T <: Real
        X = reshape(X₀ .+ 0.0, 1, length(X₀)) # This forces X to be a Matrix{AbstractFloat}        
        new{eltype(X)}(X, X' * X )
    end

end

Base.size(X::AbstractEmbedding) = size(X.X)
Base.getindex(X::AbstractEmbedding, inds...) = getindex(X.X, inds...)
function Base.:-(X::AbstractEmbedding, M::AbstractMatrix)
    X.X -= M
    G = X.X' * X.X
    return X
end
function Base.:-(X::AbstractEmbedding, b::Real)
    X.X .-= b
    G = X.X' * X.X
    return X
end
Base.:*(a::Real, X::AbstractEmbedding) = Embedding(a * X.X)
Base.:*(X::AbstractEmbedding, a::Real) = Embedding(a * X.X)

"""
    ndims(X::AbstractEmbedding)

Obtain the number of dimensinos in the Embedding X. If X is d × n,
ndims(X) returns d.

# Examples
    julia> X = Embedding(2,10);

    julia> ndims(X)
    2
"""
Base.ndims(X::AbstractEmbedding) = size(X,1)

"""
    nitems(X::AbstractEmbedding)

Obtain the number of items in the Embedding X. If X is d × n,
nitems(X) returns n.

# Examples
    julia> X = Embedding(2,10);

    julia> nitems(X)
    10
"""
nitems(X::AbstractEmbedding) = size(X,2)

"""
    J(n::Int)

Compute a normalizing matrix of dimension n. The matrix V centers the rows or columns
of a matrix that is pre or post-multiplied by V.

# Examples
    
    julia> J(10)
    10×10 Array{Float64,2}:
      0.9  -0.1  -0.1  -0.1  -0.1  -0.1  -0.1  -0.1  -0.1  -0.1
     -0.1   0.9  -0.1  -0.1  -0.1  -0.1  -0.1  -0.1  -0.1  -0.1
     -0.1  -0.1   0.9  -0.1  -0.1  -0.1  -0.1  -0.1  -0.1  -0.1
     -0.1  -0.1  -0.1   0.9  -0.1  -0.1  -0.1  -0.1  -0.1  -0.1
     -0.1  -0.1  -0.1  -0.1   0.9  -0.1  -0.1  -0.1  -0.1  -0.1
     -0.1  -0.1  -0.1  -0.1  -0.1   0.9  -0.1  -0.1  -0.1  -0.1
     -0.1  -0.1  -0.1  -0.1  -0.1  -0.1   0.9  -0.1  -0.1  -0.1
     -0.1  -0.1  -0.1  -0.1  -0.1  -0.1  -0.1   0.9  -0.1  -0.1
     -0.1  -0.1  -0.1  -0.1  -0.1  -0.1  -0.1  -0.1   0.9  -0.1
     -0.1  -0.1  -0.1  -0.1  -0.1  -0.1  -0.1  -0.1  -0.1   0.9

---

    julia> using Statistics: mean
    
    julia> A = rand(2,2); mean(A, dims=1)
    1×2 Array{Float64,2}:
     0.502544  0.538744

    julia> mean(J(2)*A, dims=1)
    1×2 Array{Float64,2}:
     0.0  0.0

    julia> mean(A*J(2), dims=2)
    2×1 Array{Float64,2}:
     0.0
     0.0

"""
J(n::Int) = Matrix(I, n, n) - ones(n,n)/n