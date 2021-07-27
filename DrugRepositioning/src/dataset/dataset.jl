struct Dataset
    drugs::Array{String}
    diseases::Array{String}
    ndrugs::Integer
    ndiseases::Integer
    associations::Matrix
    characteristics::Dict{Symbol,Array{Matrix}}
    similaritys::Dict{Symbol,Array{Matrix}}
    α::Float32
    step::Int64
end

function show(io::IO, x::Dataset)
    msg = [
            "A Dataset($(x.ndrugs) drugs, $(x.ndiseases) diseases).",
            "$(length(x.characteristics[:drug])) drug characteristics.",
            "$(length(x.characteristics[:disease])) disease characteristics.",
            "$(length(x.similaritys[:drug])) drug similaritys.",
            "$(length(x.similaritys[:disease])) disease similaritys.",
            "$(sum(x.associations)) knowns in all $(length(x.associations)) associations."
        ]
    print(io, join(msg, '\n'))
end

function get_similarity(data;dist=CosineDist,dims=2)
    1 .- pairwise(dist(), data, dims=dims)
end

function check_items(data::Array{Matrix}, n::Integer)
    all(x -> size(x, 2) == n, data)
end

function Dataset(drugs::Array{String}, diseases::Array{String}, ndrugs::Integer, ndiseases::Integer, associations::Matrix, characteritics::Dict{Symbol,Array{Matrix}}, distance::DataType, α, step)
    length(drugs) == ndrugs || throw(ArgumentError("A mismatch between `drugs` and `ndrugs`."))
    length(diseases) == ndiseases || throw(ArgumentError("A mismatch between `disease` and `ndiseases`."))
    size(associations) == (ndrugs, ndiseases) || throw(ArgumentError("`associations` must be between $ndrugs drugs and $ndiseases diseases."))
    check_items(characteritics[:drug], ndrugs) || throw(ArgumentError("Each drug characteristic must has $ndrugs drugs."))
    check_items(characteritics[:disease], ndiseases) || throw(ArgumentError("Each disease characteristic must has $ndiseases diseases."))
    similaritys = Dict{Symbol,Array{Matrix}}()
    for k in keys(characteritics)
        similaritys[k] = [get_similarity(c;dist=distance,dims=2) for c in characteritics[k]]
    end
    Dataset(drugs, diseases, ndrugs, ndiseases, associations, characteritics, similaritys, α, step)
end

function Dataset(ndrugs::Integer, ndiseases::Integer, associations::Matrix, characteristics::Dict{Symbol,Array{Matrix}};distance=CosineDist,α=0.3,step=4)
    drugs = ["drug_$i" for i in 1:ndrugs]
    diseases = ["disease_$i" for i in 1:ndiseases]
    Dataset(drugs, diseases, ndrugs, ndiseases, associations, characteristics, distance, α, step)
end

function Dataset(drugs::Array{String}, diseases::Array{String}, associations::Matrix, characteristics::Dict{Symbol,Array{Matrix}};distance=CosineDist, α=0.3, step=4)
    ndrugs = length(drugs)
    ndiseases = length(diseases)
    Dataset(drugs, diseases, ndrugs, ndiseases, associations, characteristics, distance, α, step)
end

function add_similarity!(d::Dataset, target::Symbol, similarity::Matrix)
    if target == :drug
        size(similarity) == (d.ndrugs, d.ndrugs) || throw(ArgumentError("`similarity` must have $(d.ndrugs) drugs."))
    elseif target == :disease
        size(similarity) == (d.ndiseases, d.ndiseases) || throw(ArgumentError("`similarity` must have $(d.ndiseases) diseases."))
    else
        throw(ArgumentError("'target' must be in [:drug, :disease]."))
    end
    push!(d.similaritys[target], similarity)
    d
end


struct Samples{T <: AbstractFloat}
    "用于生成样本特征的张量"
    D::Union{Array{T,3},CuArray{T,3}}
    Ds
    "用于存储真实的标签"
    label::Union{Array{Bool,2},CuArray{Bool,2}}
    "样本在标签矩阵中的位置"
    indices::Array{CartesianIndex{2}}
    ndrugs::Integer
    "疾病个数"
    ndiseases::Integer
    use_gpu::Bool
    mode::Symbol
end 

function Samples(data::Dataset, associations::Matrix, label::Matrix, pairs::Array;use_gpu::Bool=false,mode::Symbol=:left)
    size(associations) == (data.ndrugs, data.ndiseases) || throw(ArgumentError("The `associations` must be size($(d.ndrugs),$(data.ndiseases))."))
    eltype(pairs) <: Union{CartesianIndex,Integer} || throw(ArgumentError("`indices' must be `CartesianIndex`or `Integer`"))
    mode in [:left,:both,:right] || throw(ArgumentError("`mode` must be in [:left,:both,:right]."))
    D = [ Float32[r associations;associations' d] for r in data.similaritys[:drug] for d in data.similaritys[:disease]]
    Ds = ifelse(mode != :right, cat([random_walk(d, data.α, data.step) for d in D]..., dims=3), Float32[]) 
    Samples(
        ifelse(use_gpu, cat(D..., dims=3) |> gpu, cat(D..., dims=3)),
        ifelse(use_gpu, Ds |> gpu, Ds),
        label,
        ifelse(eltype(pairs) == CartesianIndex, pairs, CartesianIndices(size(label))[pairs]),
        data.ndrugs,
        data.ndiseases,
        use_gpu,
        mode
    )
end

function change_indices!(s::Samples, indices::Array{CartesianIndex{2}})
    empty!(s.indices)
    append!(s.indices, indices)
    s
end

show(io::IO,x::Samples) = print(io, "$(x.imax) samples on $(ifelse(x.use_gpu, "gpu", "cpu")).")


function get_x(s::Samples, mode::Symbol, i)
    indices = s.indices[i]
    if mode == :left
        return cat(
            [
                @view s.Ds[[index[1],index[2] + s.ndrugs],:,:,:] for index in indices
            ]...;
            dims=ndims(s.Ds) + 1
        )
    elseif mode == :right
        return cat(
            [
                @view s.D[[index[1],index[2] + s.ndrugs],:,:] for index in indices
            ]...;
            dims=ndims(s.D) + 1
        )
    else
        return get_x(s, :left, i), get_x(s, :right, i)
    end
end

function get_y(s::Samples, i)
    indices = s.indices[i]
    Flux.onehotbatch(s.label[indices], [false,true])    
end

function _getobs(s::Samples, i)
    indices = s.indices[i]
    x = get_x(s, s.mode, i)
    y = get_y(s, i)
    indices, x, ifelse(s.use_gpu, gpu(y), y)
end

_nobs(s::Samples) = length(s.indices)
export Dataset,Samples,add_similarity!,change_indices!