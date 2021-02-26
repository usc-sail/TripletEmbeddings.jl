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