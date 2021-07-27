using DelimitedFiles:readdlm
using Random:shuffle!
using Flux
using Plots
using CUDA

using DrugRepositioning

CUDA.allowscalar(false)

"原始数据存储路径"
data_dir=joinpath(@__DIR__,"data","origin")

"对比方法测试结果"
compare_dir=joinpath(@__DIR__,"data","all_result")
ms=compared_scores(compare_dir)

"高玲的方法的测试结果"
gao_dir=joinpath(@__DIR__,"data","all_result","gao")
P=readdlm(joinpath(gao_dir,"2","0mean5_P.txt"))[:]
FPR=readdlm(joinpath(gao_dir,"2","0mean5_FPR.txt"))[:]
TPR=readdlm(joinpath(gao_dir,"2","0mean5_TPR.txt"))[:]
R=TPR
pushfirst!(ms,MetricsScore("GPred_2",R,P,FPR,TPR,[],[]))
P=readdlm(joinpath(gao_dir,"1","0mean5_P.txt"))[:]
FPR=readdlm(joinpath(gao_dir,"1","0mean5_FPR.txt"))[:]
TPR=readdlm(joinpath(gao_dir,"1","0mean5_TPR.txt"))[:]
R=TPR
pushfirst!(ms,MetricsScore("GPred_1",R,P,FPR,TPR,[],[]))

"读取CSV"
function read_csv(file_path;d_type::DataType=Float64,i_type::DataType=String,h_dtype::DataType=String)
    f=readdlm(file_path,',')
    header=f[1,2:end]|>Array{String}
    indexer=f[2:end,1]|>Array{String}
    data=f[2:end,2:end]|>Array{d_type}
    indexer,header,data
end

"药物疾病关联"
function drug_disease()
    return read_csv(joinpath(data_dir,"drug_disease_mat.csv");d_type=Bool)
end

"疾病相似性"
function disease_similarity()
    return read_csv(joinpath(data_dir,"disease_similarity_mat.csv");d_type=Float32)
end

"药物靶标域"
drug_domain()=read_csv(joinpath(data_dir,"drug_target_domain_mat.csv");d_type=Bool)
"药物靶标GO"
drug_go()=read_csv(joinpath(data_dir,"drug_target_go_mat.csv");d_type=Bool)
"药物公共化学子结构"
drug_pubchem()=read_csv(joinpath(data_dir,"drug_pubchem_mat.csv");d_type=Bool)


drugs,diseases,RD=drug_disease()
_,_,D1=disease_similarity()
_,_,RP1=drug_domain()
_,_,RP2=drug_go()
_,_,RP3=drug_pubchem()
_,_,RP4=drug_disease()

props=Dict{Symbol,Array{Matrix}}(
    :drug=>[RP1',RP2',RP3',RP4'],
    :disease=>[]
)

dataset=Dataset(drugs,diseases,RD,props)
add_similarity!(dataset,:disease,D1)
model=CombineModel(dataset;feature_left=128,feature_right=512)|>gpu
# trainer=Trainer(model,Flux.Losses.logitcrossentropy,dataset;epochs=80)
# training(trainer,Flux.Optimise.ADAM(3e-4))
res=ValidateResult("result")
df=analyze_by_drugs(res)
R,P,FPR,TPR=pr_roc(res)
my_score=MetricsScore("MSRD",R,P,FPR,TPR,[],[])
pushfirst!(ms,my_score)
pr_roc_curve(ms)|>display
plot(res)
