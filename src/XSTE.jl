struct XSTE <: AbstractLoss
    loss::Function
    grad::Function

    function XSTE(;σ = 1/√(2), lossfunction = LogitMarginLoss(), distance = SqEuclidean())
        function kernel(X, t)
            @views D = pairwise(distance, X[:, [t.i, t.j, t.k]], dims=2)
            return (D[1,3] - D[1,2])/(2σ^2)
        end
        loss(X, t::Triplet) = value(lossfunction, kernel(X, t))
        grad(X, t::Triplet) = Zygote.gradient(X -> loss(X, Zygote.dropgrad(t)), X)[1]

        new(loss, grad)
    end
end

function gradient(loss::XSTE, triplets::Triplets, X::AbstractMatrix)
    X = sparse(X)
    @inbounds C = ThreadsX.sum(loss.loss(X, t) for t in triplets)
    @inbounds ∇C = ThreadsX.sum(loss.grad(X, t) for t in triplets)
    return C, ∇C
end