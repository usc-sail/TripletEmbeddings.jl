struct LinearTransformation{T}
    s::T         # Scaling
    R::Matrix{T} # Rotation
    b::Vector{T} # Bias

    function LinearTransformation(s::T, R::Matrix{T}, b::Vector{T}) where T <: Real
        new{T}(s, R, b)
    end
end

"""
    apply(tr::LinearTransformation, X::Matrix{T}) where T <: Real

Apply a linear transformation (scaling, rotation, and bias) to the matrix X.
"""
function apply(tr::LinearTransformation, X::Matrix{T}) where T <: Real
    Xᶜ = X * J(size(X, 2)) # Centers the (columns of the) matrix
    return broadcast(+, tr.s * tr.R * Xᶜ, tr.b)
end

"""
    apply(tr::LinearTransformation, X::AbstractEmbedding{T}) where T <: Real

Apply a linear transformation (scaling, rotation, and bias) to the Embedding X.
"""
function apply(tr::LinearTransformation, X::AbstractEmbedding{T}) where T <: Real
    return Embedding(apply(tr, X.X))
end

"""
    procrustes(X::AbstractEmbedding{T}, Y::Matrix{T}) where T <: Real

Compute the optimal scaling, rotation, and translation for X to match Y in the least
squares sense. X should be the same size as Y.

The algorithm to find the optimal rotation and translation is also known as Kabsch algorithm.

---

    procrustes(X::AbstractMatrix{T}, Y::Matrix{T}) where T <: Real

Compute the optimal scaling, rotation, and translation for X to match Y in the least
squares sense. X should be the same size as Y.

The algorithm to find the optimal rotation and translation is also known as Kabsch algorithm.

# References
    
    1. Dokmanic, I., Parhizkar, R., Ranieri, J., & Vetterli, M. (2015). Euclidean 
       distance matrices: essential theory, algorithms, and applications.
       IEEE Signal Processing Magazine, 32(6), 12-30.
"""
function procrustes(X::AbstractEmbedding{T}, Y::Matrix{T}) where T <: Real
    # @assert nitems(X) > 1 "nitems(X) must be > 1 to use procrustes"
    @assert ndims(X) == size(Y,1)  "ndims(X) must be equal to size(Y,2)"

    X.X, _ = procrustes(X.X, Y)
    X.G = X.X' * X.X

    return X
end

function procrustes!(X::AbstractEmbedding{T}, Y::Matrix{T}) where T <: Real
    X, _ = procrustes(X, Y)
end

function procrustes(X::Matrix{T}, Y::Matrix{T}) where T <: Real
    # We *could* implement this for size(X) != size(Y) if needed...
    @assert size(X) == size(Y) "X and Y must be the same size"

    # Find the number of columns
    n = size(X,2) # Number of columns in X
    m = size(Y,2) # Number of columns in Y

    # Center both embeddings
    # The subscript for c is not defined in unicode :-(, so we use superscripts
    Xᶜ = X * J(n) # Centers the (columns of the) Embedding
    Yᶜ = Y * J(m) # Centers the (columns of the) Embedding

    # We use the SVD to find the optimal rotation matrix
    U, S, V = svd(Xᶜ * Yᶜ')
    R = V * U' # Optimal rotation matrix

    # We can now find the best scaling, after the embeddings are centered and
    # have the same orientation. The scaling is found by solving min_s ||s * X - Y||_F^2
    yᶜ = Y * ones(m)/m
    s = norm(Yᶜ) / norm(Xᶜ)

    transform = LinearTransformation(s, R, yᶜ)
    
    # We finally scale, rotate, and translate our embedding to match Y as best as possible
    return apply(transform, X), transform
end

function mse(X::AbstractEmbedding{T}, Y::Matrix{T}) where T <: Real
    return norm(X.X - Y)^2/(nitems(X) * ndims(X))
end