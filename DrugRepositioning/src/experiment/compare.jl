struct MetricsScore
    name::String
    r::Array{Float64}
    p::Array{Float64}
    fpr::Array{Float64}
    tpr::Array{Float64}
    aupr::Array{Float64}
    auroc::Array{Float64}
end

show(io::IO,x::MetricsScore)=print(io,"The metrics of $(x.name)")

function compared_scores(data_dir::String)
    names=[
        "CBPred",
        "LRSSL ",
        "SCMFDD",
        "HGBI  ",
        "MBiRW "
    ]
    scores=MetricsScore[]
    for (i,name) in enumerate(names)
        P=readdlm(joinpath(data_dir,"$(i)mean5_P.txt"))[:]
        FPR=readdlm(joinpath(data_dir,"$(i)mean5_FPR.txt"))[:]
        TPR=readdlm(joinpath(data_dir,"$(i)mean5_TPR.txt"))[:]
        R=TPR
        AUROC=readdlm(joinpath(data_dir,"$(i)auc.txt"))[:]
        AUPR=readdlm(joinpath(data_dir,"$(i)aupr.txt"))[:]
        push!(scores,MetricsScore(name,R,P,FPR,TPR,AUPR,AUROC))
    end
    scores
end

function pr_roc_curve(scores::Array{MetricsScore})
    curve_roc=plot(title="ROC",xlabel="Recall",ylabel="Precision")
    curve_pr=plot(title="PR",xlabel="FPR",ylabel="TPR")
    for s in scores
        n=s.name
        au=auc(s.r,s.p)
        plot!(curve_pr,s.r,s.p,label="$n $(round(au,digits=3))")
        au=auc(s.fpr,s.tpr)
        plot!(curve_roc,s.fpr,s.tpr,label="$n $(round(au,digits=3))")
    end
    plot(curve_pr,curve_roc,layout=(1,2))
end

function topk_recall_curve(scores::Array{MetricsScore},ks::StepRange)
    666
end


export compared_scores,MetricsScore,pr_roc_curve,topk_recall_curve