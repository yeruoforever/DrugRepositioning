struct CrossValidation
    indices
    partition::Integer
    samples::Array
    folds::Integer
    use_all::Bool
end

function CrossValidation(folds::Integer, samples::AbstractArray;shuffle::Bool=true,use_all::Bool=true)
    ids = LinearIndices(samples) |> collect
    shuffle!(ids)
    partition = length(ids) ÷ folds
    CrossValidation(ids, partition, samples, folds, use_all)
end

function (x::CrossValidation)(fold::Integer;)
    (fold < 1 || fold > x.folds) && throw(ArgumentError("Please check 1 ≤ fold ≤ $(x.folds)"))
    last = x.use_all ? length(x.indices) : x.folds * x.partition
    select_start = (fold - 1) * x.partition + 1
    select_end = fold * x.partition
    train = x.samples[x.indices[1:select_start - 1]]
    append!(train, x.samples[x.indices[select_end + 1:last]])
    test = x.samples[x.indices[select_start:select_end]]
    return (train = train, test = test)
end


struct ValidateResult{T <: AbstractFloat}
    dir::String
    scores::Matrix{T}
    labels::Matrix{Bool}
    mask::Matrix{Bool}
    drugs::Array{String}
    diseases::Array{String}
end

show(io::IO,vr::ValidateResult) = print(io, "Validate Results($(vr.dir))")

function ValidateResult(dir::String)
    labels = readdlm(joinpath(dir, "RD.csv"), ',', Bool)
    drugs = readdlm(joinpath(dir, "drugs.csv"), ',', String)
    diseases = readdlm(joinpath(dir, "diseases.csv"), ',', String)
    mask = zeros(size(labels))
    scores = zeros(Float32, size(labels))
    folds = map(x->joinpath(abspath(dir),x),readdir(dir))
    filter!(isdir,folds)
    for f in folds
        mask = mask .+ readdlm(joinpath(dir, f, "mask.csv"), ',', Bool)
        scores = scores .+ readdlm(joinpath(dir, f, "result.csv"), ',', Float32)
    end
    scores = scores ./ mask
    ValidateResult(dir, scores, labels, map(>(0), mask), drugs, diseases)
end

function analyze(res::ValidateResult;union_all::Bool=false)
    if union_all
        cms = threshold_scan(res.labels[res.mask], res.scores[res.mask])
    else
        cms_all = []
        for (name, l, s, m) in zip(res.drugs, eachrow(res.labels), eachrow(res.scores), eachrow(res.mask))
            sum(m)==0 && continue
            tmp = threshold_scan(l[m], s[m])
            push!(cms_all, tmp)
        end
        cms=[]
        for i=1:length(cms_all[1])
            cm=ConfusionMatrix(zeros(eltype(cms_all[1][1]),size(cms_all[1][1])))
            for each in cms_all
                length(each)<i && continue
                cm=cm+each[i]
            end
            push!(cms,cm)
        end
    end
    cms
end

function roc(r::ValidateResult;union_all::Bool=false)
    cms=analyze(r,union_all=union_all)
    TPR = tpr.(cms, 2)
    FPR = fpr.(cms, 2)
    FPR,TPR
end

function pr(r::ValidateResult,union_all::Bool=false)
    cms=analyze(r,union_all=union_all)
    TPR = tpr.(cms, 2)
    FPR = fpr.(cms, 2)
    FPR,TPR
end

function analyze_by_drugs(result::ValidateResult)
    df = DataFrame("Drug Name" => String[], "AUPR" => Float64[], "AUROC" => Float64[])
    for (name, l, s, m) in zip(result.drugs, eachrow(result.labels), eachrow(result.scores), eachrow(result.mask))
        sum(m)==0 && continue
        cms = threshold_scan(l[m], s[m])
        TPR = tpr.(cms, 2)
        FPR = fpr.(cms, 2)
        P = precision.(cms, 2)
        R = recall.(cms, 2)
        aupr = auc(R, P)
        auroc = auc(FPR, TPR)
        push!(df, (name, aupr, auroc))
    end
    push!(df, ("Average of $(size(df, 1)) drugs", mean(df.AUPR), mean(df.AUROC)))
    df
end


function pr_roc(res::ValidateResult;union_all::Bool=false)
    cms=analyze(res;union_all=union_all)
    TPR = tpr.(cms, 2)
    FPR = fpr.(cms, 2)
    P = precision.(cms, 2)
    R = recall.(cms, 2)
    R,P,FPR,TPR
end

function plot(res::ValidateResult;union_all::Bool=false,)
    R,P,FPR,TPR=pr_roc(res,union_all=union_all)
    aupr = auc(R, P)
    auroc = auc(FPR, TPR)
    left = plot(R, P, title="P-R(auc=$(round(aupr, digits=3)))")
    right = plot(FPR, TPR, title="ROC(auc=$(round(auroc, digits=3)))")
    plot(left, right, layout=(1, 2)) |> display
end

export CrossValidation,ValidateResult,analyze_by_drugs,analyze,pr,roc,pr_roc