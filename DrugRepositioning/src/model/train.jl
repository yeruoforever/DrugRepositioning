mutable struct Config{T <: AbstractFloat}
    folds::Integer      
    epochs::Integer
    bs::Integer
    lr::T
end

show(io::IO,c::Config) = print(io, "A trainer config.\n(epochs $(c.epochs), batch size $(c.bs), learn rate $(c.lr)).\nThere are $(c.folds) folds.")

mutable struct Status
    epoch::Integer
    loss::Dict{String,Array{Float64}}
end

function reset_model!(model;init=Flux.glorot_uniform)
    Flux.loadparams!(model, map(x -> init(size(x)), Flux.params(model)))
    model
end

struct Trainer 
    model
    loss::Function
    data::Dataset
    folds::Int
    epochs::Int
    bs::Int
end

Trainer(model,loss,data;folds=5,epochs=100,batch_size=16) = Trainer(model, loss, data, folds, epochs, batch_size)

function training(trainer::Trainer, opt;log_dir::String="./result",use_gpu=true)
    RD = trainer.data.associations
    knowns = findall(RD .== 1)
    unknowns = findall(RD .== 0)
    shuffle!(knowns)
    shuffle!(unknowns)
    selected = eachindex(knowns)
    cv = CrossValidation(trainer.folds, selected)
    ps = params(trainer.model)
    plot_train = plot()
    plot_validate = plot()
    isdir("./result") || mkdir("./result")
    writedlm("./result/RD.csv", RD, ',')
    writedlm("./result/drugs.csv", trainer.data.drugs, ',')
    writedlm("./result/diseases.csv", trainer.data.diseases, ',')
    for f in 1:trainer.folds
        @info "fold $f"
        rd = copy(RD)
        loss_train = []
        loss_val = []
        train, test = cv(f)
        rd[knowns[test]] .= false
        mask = falses(size(RD))
        mask[knowns[test]] .= true
        useful = map(>=(1), sum(mask, dims=2) |> vec)
        mask = trues(size(RD))
        mask[knowns[train]] .= false
        mask[unknowns[train]] .= false
        reset_model!(trainer.model)
        samples = Samples(trainer.data, rd, RD, findall(!, mask);use_gpu=use_gpu,mode=:both)
        @info "training..."
        for e in 1:trainer.epochs
            loss_epoch = []
            val_epoch = []
            change_indices!(samples, findall(!, mask)) 
            loader = Flux.Data.DataLoader(samples, batchsize=trainer.bs)
            for (i, (x1, x2), y) in loader
                local loss
                gs = gradient(ps) do
                    ŷ = trainer.model(x1, x2)
                    loss = trainer.loss(ŷ, y)
                end
                Flux.update!(opt, ps, gs)
                push!(loss_epoch, loss)
            end
            push!(loss_train, mean(loss_epoch))
            validate = map(>=(4), sum(mask, dims=2) |> vec)
            change_indices!(samples, findall(mask .& validate))
            loader = Flux.Data.DataLoader(samples, batchsize=512)
            for (i, (x1, x2), y) in loader
                ŷ = trainer.model(x1, x2)
                loss = trainer.loss(ŷ, y)
                push!(val_epoch, loss)
            end
            push!(loss_val, mean(val_epoch))
        end
        plot!(plot_train, loss_train, label="fold $f")
        plot!(plot_validate, loss_val, label="fold $f")
        plot(plot_train,plot_validate,layout=(2,1))|>display
        @info "testing..."
        @info "$(sum(useful)) drugs to test."
        mask = mask .& useful
        @info "total $(sum(mask)) drug-disease pairs to test."
        change_indices!(samples, findall(mask))
        loader = Flux.Data.DataLoader(samples, batchsize=1024)
        result = zeros(size(RD))
        for (i, x, y) in loader
            ŷ = trainer.model(x...)
            ȳ = cpu(ŷ)
            result[i] .= softmax(ȳ)[2,:]
        end
        h = heatmap(result[useful,:])
        yticks!(h, [1:length(useful);], trainer.data.drugs[useful])
        display(h)
        isdir("./result/fold_$f") || mkdir("./result/fold_$f")
        writedlm("./result/fold_$f/result.csv", result, ',')
        writedlm("./result/fold_$f/mask.csv", mask, ',')
    end
end

export Config,Status,reset_model!,Trainer,training