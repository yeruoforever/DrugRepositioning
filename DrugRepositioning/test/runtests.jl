push!(LOAD_PATH,".")

using DelimitedFiles
using CUDA
using Zygote
using Flux
using BSON
using Test
using Plots

using DrugRepositioning

data = readdlm(raw"E:\Projects\DrugRepositioning\data\origin\drug_disease_mat.csv", ',')
drugs = data[2:end,1] |> Array{String}
diseases = data[1,2:end] |> Array{String}
RD = data[2:end,2:end] |> Array{Bool}
data = readdlm(raw"E:\Projects\DrugRepositioning\data/origin/drug_pubchem_mat.csv", ',')
dp1 = data[2:end,2:end] |> Array{Float32}
data = readdlm(raw"E:\Projects\DrugRepositioning\data/origin/drug_target_go_mat.csv", ',')
dp2 = data[2:end,2:end] |> Array{Float32}
chara = Dict{Symbol,Array{Matrix}}(:drug => [dp1',dp2'], :disease => [])
data = readdlm(raw"E:\Projects\DrugRepositioning\data/origin/disease_similarity_mat.csv", ',')
sim = data[2:end,2:end] |> Array{Float32}

@testset "DrugRepositioning" begin    
    local dataset
    local s
    @testset "dataset" begin
        sim=DrugRepositioning.get_similarity(RD)
        heatmap(sim)|>display
        @test true
        dataset = Dataset(drugs, diseases, RD, chara;α=0.3,step=4)
        @test true
        n = length(dataset.similaritys[:disease])
        add_similarity!(dataset, :disease, sim)
        @test length(dataset.similaritys[:disease]) - n == 1
        @test length(keys(dataset.similaritys))==2
        @test length(keys(dataset.similaritys[:drug]))==2
        @test length(keys(dataset.similaritys[:disease]))==1
        s = Samples(dataset, RD, RD, findall(RD .== 1);mode=:both)
        @test true
        s = Samples(dataset, RD, RD, findall(RD .== 1);use_gpu=true,mode=:both)
        @test true
        change_indices!(s,findall(RD.==0))
        @test Flux.Data._nobs(s)==sum(RD.==0)
    end
    @testset "experiment" begin
        labels = findall(RD .== 1)
        maxn = rand(3:7)
        n_folds = CrossValidation(maxn, labels, shuffle=true)
        @test true
        n = rand(1:maxn)
        train, test = n_folds(n)
        @test length(train) > length(test)
        @test length(train) + length(test) == length(labels)
        vr=ValidateResult(raw"E:\Projects\DrugRepositioning\result")
        @test true
        plot(vr)|>display
        @test true
        # println(vr)
        r=analyze(vr)
        @test true
        # println(r)
        ms=compared_scores(raw"E:\Projects\DrugRepositioning\data/all_result")
        pr_roc_curve(ms)|>display
        @test true
    end
    @testset "model" begin
        att = Attention(10)
        @test size(att(rand(Float32,10,10)))==(10,10)
        att2=Attention(10,1)
        @test size(att2(rand(Float32,10,10)))==(1,10)
        att3=AttrAttention(2888,500,3)|>gpu
        @test size(att3(CUDA.rand(Float32,2,1444,3,7,9)))==(500,7,9)
        att4=ScaleAttention(500)
        @test size(att4(rand(Float32,500,7,9)))==(500,7,9)
        x1 = rand(Float32, 2, 1444, 2, 16)
        m1 = RDConvolution(dataset,out=2)
        @test true 
        ŷ1 = m1(x1)
        @test true
        @test size(ŷ1) == (2, 16)
        x2 = CUDA.rand(Float32, 2, 1444, 2, 16)
        m2 = RDConvolution(dataset;out=1024) |> Flux.gpu
        @test true
        ŷ2 = m2(x2)
        @test true
        @test size(ŷ2) == (1024, 16)
        DrugRepositioning.save("./weights.bson", m2)
        weights = DrugRepositioning.load("./weights.bson")
        m2 = RDConvolution(dataset) |> Flux.gpu
        Flux.loadparams!(m2, weights)
        ŷ3 = m2(x2)
        @test all(ŷ3 .== ŷ2)
        rm("./weights.bson")
        reset_model!(m2, init=Flux.kaiming_uniform)
        ŷ = m2(x2)
        @test true
        loader = Flux.Data.DataLoader(s, batchsize=3)
        for (i, (x1,x2), y) in loader
            ŷ = m2(x2)
            break
        end
        @test true
        m3=MSRD(dataset)|>gpu
        @test size(m3.attratt.encoders,3)==2
        for(i,(x1,x2),y) in loader
            ŷ=m3(x1)
            break
        end
        @test true
        m4=CombineModel(dataset)|>gpu
        for (i,(x1,x2),y) in loader
            ŷ=m4(x1,x2)
            @test size(ŷ)==(2,3)
            break
        end
        @test true
        # trainer = Trainer(m2, Flux.Losses.crossentropy, dataset, epochs=10)
        # training(trainer, Flux.Optimise.Descent())
        # @test true
    end
end