function plot(X::Embedding{T}, kwargs...) where T
    d, n = size(X)

    if d == 1
        plot(X', kwargs...)
    elseif
        d == 2
        plot(X[:,1], X[:,2])
    end
end

function plot!(X::Embedding{T}, kwargs...) where T
    d, n = size(X)

    if d == 1
        plot!(X, kwargs...)
    elseif
        d == 2
        plot!(X[:,1], X[:,2], kwargs...)
    end
end