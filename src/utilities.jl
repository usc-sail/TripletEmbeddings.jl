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
function scale(X::AbstractEmbedding{T}, Y::AbstractMatrix{T}; MSE::Bool=false) where T <: Real
    # We solve the scaling problem by min || aX - Y - b||^2,
    # where (a,b) are the scale and offset parameters
    @assert ndims(X) == 1 "ndims(X) > ndims(Y)"
    @assert length(Y) == nitems(X) "Data needs to have the same number of elements as X"

    a, b = [X.X' -ones(size(X.X'))]\Y'

    if MSE
        return a*X - b, norm(a*X - b .- Y)^2
    else
        return a*X - b
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
    # We solve the scaling problem by min || aX - Y - b||^2,
    # where (a,b) are the scale and offset parameters
    @assert ndims(X) == 1 "ndims(X) > ndims(Y)"
    @assert length(Y) == nitems(X) "Data needs to have the same number of elements as X"

    a, b = [X.X' -ones(size(X.X'))]\Y'

    X.X = a*X.X .- b
    X.G = X.X' * X.X
    return
end

"""
    procrustes(X::AbstractEmbedding{T}, Y::AbstractMatrix{T}) where T <: Real

Compute the optimal scaling, rotation, and translation for X to match Y in the least
squares sense. X should be the same size as Y.

The algorithm to find the optimal rotation and translation is also known as Kabsch algorithm.

# References
    
    1. Dokmanic, I., Parhizkar, R., Ranieri, J., & Vetterli, M. (2015). Euclidean 
       distance matrices: essential theory, algorithms, and applications.
       IEEE Signal Processing Magazine, 32(6), 12-30.
"""
function procrustes(X::AbstractEmbedding{T}, Y::AbstractMatrix{T}) where T <: Real
    # We *could* implement this for size(X) != size(Y) if needed...
    @assert size(X) == size(Y) "X and Y must be the same size"

    # Find the number of columns
    n = nitems(X)
    m = size(Y,2) # Number of columns in Y

    # Center both embeddings
    # The subscript for c is not defined in unicode :-(, so we use superscripts
    Xᶜ = X * J(nitems(X)) # Centers the (columns of the) Embedding
    Yᶜ = Y * J(nitems(X)) # Centers the (columns of the) Embedding

    # We use the SVD to find the optimal rotation matrix
    U, S, V = svd(Xᶜ * Yᶜ')
    R = V * U' # Optimal rotation matrix

    # We can now find the best scaling, after the embeddings are centered and
    # have the same orientation. The scaling is found by solving min_s ||s * X - Y||_F^2
    yᶜ = Y * ones(m)/m
    s = norm(Yᶜ) / norm(Xᶜ)
    
    # We finally scale, rotate, and translate our embedding to match Y as best as possible
    X.X = broadcast(+, s * R * Xᶜ, yᶜ)
    X.G = X.X' * X.X

    return X

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