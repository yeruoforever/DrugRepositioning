function random_walk(X::AbstractMatrix,α,step)
    X̄=X./sum(X,dims=2)
    Xs=Matrix{Float32}[X̄]
    for i=2:step
        push!(Xs,Xs[end]*X̄+α*X̄)
    end
    cat(Xs...,dims=4)
end