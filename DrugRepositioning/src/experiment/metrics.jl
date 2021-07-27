struct ConfusionMatrix{T<:Integer}
    c::Array{T,2}
end

size(c::ConfusionMatrix,kws...)=size(c.c,kws...)
sum(c::ConfusionMatrix,kw...)=sum(c.c,kw...)
+(a::ConfusionMatrix,b::ConfusionMatrix)=ConfusionMatrix(a.c.+b.c)
eltype(a::ConfusionMatrix)=eltype(a.c)

function confusion(labels, predicts, classes)
    predict_real = zeros(Integer, classes, classes)
    for (p, r) in zip(predicts, labels)
        predict_real[p + 1,r + 1] += 1
    end
    return ConfusionMatrix(predict_real)
end


tp(c::ConfusionMatrix,t) = c.c[t,t]
fp(c::ConfusionMatrix,t) = sum(c.c[t,1:t - 1]) + sum(c.c[t,t + 1:end])
tn(c::ConfusionMatrix,t) = sum(c.c[1:t - 1,1:t - 1]) + sum(c.c[1:t - 1,t + 1:end]) + sum(c.c[t + 1:end,1:t - 1]) + sum(c.c[1 + t:end,1 + t:end])
fn(c::ConfusionMatrix,t) = sum(c.c[1:t - 1,t]) + sum(c.c[t + 1:end,t])

function tpr(c::ConfusionMatrix, t=0)
    if t == 0
        cs = size(c, 1)
        TP = [tp(c, i) for i in 1:cs]
        FN = [fp(c, i) for i in 1:cs]
    else
        TP = tp(c, t)
        FN = fn(c, t)
    end
    return TP ./ (TP .+ FN)
end


function fpr(c::ConfusionMatrix, t=0)
    if t == 0
        cs = size(c, 1)
        FP = [fp(c, i) for i in 1:cs]
        TN = [tn(c, i) for i in 1:cs]
    else
        FP = fp(c, t)
        TN = tn(c, t)
    end
    return FP ./ (FP .+ TN)
end

recall = tpr
sensitivity = tpr

function precision(c::ConfusionMatrix, t=0)
    if t == 0
        cs = size(c, 1)
        TP = [tp(c, i) for i in 1:cs]
        FP = [fp(c, i) for i in 1:cs]
    else
        TP = tp(c, t)
        FP = fp(c, t)
    end
    return TP ./ (TP .+ FP)
end

function accuracy(c::ConfusionMatrix, t=0)
    if t == 0
        cs = size(c, 1)
        TP = [tp(c, i) for i in 1:cs]
        TN = [tn(c, i) for i in 1:cs]
    else
        TP = tp(c, t)
        TN = tn(c, t)
    end
    return (TP .+ TN) ./ sum(c)
end

function threshold_scan(labels, scores)
    len=length(labels)
    p = sortperm(scores)
    c = zeros(Int64, 2, 2)
    tn, fp, fn, tp = CartesianIndices(c)
    T = sum(labels)
    N = length(labels) - T
    c[tp] = T
    c[fp] = N
    result=Array{ConfusionMatrix,1}(undef,len-1)
    for (j,i) in enumerate(p) 
        if labels[i] == true
            c[fn] += 1
            c[tp] -= 1
        else
            c[tn] += 1
            c[fp] -= 1
        end
        j == len && break
        result[j]=ConfusionMatrix(copy(c))
    end
    result
end



function auc(x::Array{<:Real}, y::Array{<:Real})
    n = length(x)
    length(y) ≠ n && throw(DimensionMismatch("`x` and `y` must have same length (`x` is $n, `y` is $(length(y)))."))
    issorted(x) || issorted(x, rev=true) || throw(ArgumentError("`x` must be sorted in increasing order."))
    rev = zero(promote_type(eltype(x), eltype(y)))
    p = sortperm(x)
    x₀=0
    y₀=y[p[1]]
    for i = 1:n
        Δx = x[p[i]] - x₀
        ȳ = (y[p[i]] + y₀) / 2
        if !isnan(Δx) && !isnan(ȳ)
            rev += Δx * ȳ
        end
        x₀=x[p[i]]
        y₀=y[p[i]]
    end
    rev
end

export threshold_scan,auc,tpr,fpr,recall,precision,confusion,ConfusionMatrix,accuracy