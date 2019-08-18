function partition(ntriplets::Int64, nthreads::Int64)
    
    ls = range(1, stop=ntriplets, length=nthreads+1)
    
    map(1:nthreads) do i
        a = round(Int64, ls[i])
        if i > 1
            a += 1
        end
        b = round(Int64, ls[i+1])
        a:b
    end
end

###############################################
# Scaling for 1D embeddings (LS minimization) #
###############################################

"""
    scale(X::AbstractEmbedding{T}, Y::Matrix{T}; MSE::Bool=false) where T <: Real

Scale the embedding X according to the Y, such that the MSE between both is minimized.

# Examples

    julia> Y = rand(1,100)
    julia> triplets = Triplets(Y)
    julia> X = Embedding(size(Y))
    julia> fit!(STE(), triplets, X)
    julia> X = scale(X, Y)
    julia> plot(Y')
    julia> plot(X') # Plot after rescaling X to Y
"""
function scale(X::AbstractVector{T}, Y::AbstractVector{T}) where T
    # We solve the scaling problem by min || aX - Y - b||^2,
    # where (a,b) are the scale and offset parameters
    @assert length(X) == length(Y) "Vector X and Y must be the same length"
    
    a, b = [X -ones(size(X))]\Y
    return a * X .- b
end

function scale(X::AbstractEmbedding{T}, Y::AbstractMatrix{T}; MSE::Bool=false) where T <: Real
    # This function calls scale(X::Vector{T}, Y::Vector{T}) where T
    @assert ndims(X) == 1 "ndims(X) > 1. Use procrustes instead."
    @assert ndims(X) == size(Y,1) "Dimension mismatch: ndims(X) != ndims(Y)"
    @assert length(Y) == nitems(X) "Dimension mismatch: nitems(X) != nitems(Y)"

    # We try to flatte
    Xv = dropdims(X.X', dims=2)
    Yv = dropdims(Y', dims=2)

    # Embedding accepts a vector, that then reshapes into an 1 × n embedding
    Z = Embedding(scale(Xv, Yv))

    if MSE
        return Z, norm(Z.X - Y)^2
    else
        return Z
    end
end

"""
    scale!(X::AbstractEmbedding{T}, Y::AbstractMatrix{T}) where T <: Real

Scale the embedding X according to the Y, such that the MSE between both is minimized.

# Examples

    julia> Y = rand(1,100)
    julia> triplets = Triplets(Y)
    julia> X = Embedding(size(Y))
    julia> fit!(STE(), triplets, X)
    julia> scale!(X, Y)
    julia> plot(Y')
    julia> plot(X') # Plot after rescaling X to Y

"""
function scale!(X::AbstractEmbedding{T}, Y::AbstractMatrix{T}) where T <: Real
    X = scale(X, Y)
end

"""
    procrustes(X::AbstractEmbedding{T}, Y::AbstractMatrix{T}) where T <: Real

Compute the optimal scaling, rotation, and translation for X to match Y in the least
squares sense. X should be the same size as Y.

The algorithm to find the optimal rotation and translation is also known as Kabsch algorithm.

---

    procrustes(X::AbstractMatrix{T}, Y::AbstractMatrix{T}) where T <: Real

Compute the optimal scaling, rotation, and translation for X to match Y in the least
squares sense. X should be the same size as Y.

The algorithm to find the optimal rotation and translation is also known as Kabsch algorithm.

# References
    
    1. Dokmanic, I., Parhizkar, R., Ranieri, J., & Vetterli, M. (2015). Euclidean 
       distance matrices: essential theory, algorithms, and applications.
       IEEE Signal Processing Magazine, 32(6), 12-30.
"""
function procrustes(X::AbstractEmbedding{T}, Y::AbstractMatrix{T}) where T <: Real
    @assert nitems(X) > 1 "nitems(X) must be > 1 to use procrustes"
    @assert ndims(X) == size(Y,2)  "ndims(X) must be equal to size(Y,2)"

    X.X = procrustes(X.X, Y)
    X.G = X.X' * X.X

    return X
end

function procrustes(X::AbstractMatrix{T}, Y::AbstractMatrix{T}) where T <: Real
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
    
    # We finally scale, rotate, and translate our embedding to match Y as best as possible
    return broadcast(+, s * R * Xᶜ, yᶜ)
end


function procrustes!(X::AbstractMatrix{T}, Y::AbstractMatrix{T}) where T <: Real
    X = procrustes(X, Y)
end

function procrustes!(X::AbstractEmbedding{T}, Y::AbstractMatrix{T}) where T <: Real
    X = procrustes(X, Y)
end

function findscaling(X::AbstractMatrix{T}, Y::AbstractMatrix{T}) where T <: Real
    # Define the scaling variable
    s = Variable(1)
    problem = minimize(sumsquares(s*X - Y))
    solve!(problem, SCSSolver(verbose=false))

    return s.value
end