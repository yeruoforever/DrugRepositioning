function RDConvolution(d::Dataset;out=1024,σ=NNlib.relu)
    in_channels = prod(length ∘ values, values(d.similaritys))
    encoder = Flux.Chain(
        Conv((2, 2), in_channels => 16, σ, stride=(1, 1), pad=(1, 1)),
        MaxPool((2,2),pad=(1,1),stride=(2,2)),
        # Conv((2, 2), 16 => 16, σ, stride=(2, 2), pad=(1, 1)),
        Conv((2, 2), 16 => 32, σ, stride=(1, 1), pad=(1, 1)),
        # Conv((2, 2), 32 => 32, σ, stride=(2, 2), pad=(1, 1))
        MaxPool((2,2),pad=(1,1),stride=(2,2))
    )
    outsize = Flux.outdims(encoder, (2, d.ndiseases + d.ndrugs, in_channels,1))
    classifier = Flux.Chain(
        Flux.flatten,
        Dense(prod(outsize),out,σ),
    )
    Flux.Chain(
        encoder,
        classifier
    )
end

"注意力机制"
struct Attention
    W
    b
    H
end

""
Attention(in::Integer,out::Integer=in;σ=tanh,hid::Integer=in÷2,init=Flux.glorot_uniform)=Attention(
    init(hid,in),
    init(hid,),
    init(out,hid)
)

function (a::Attention)(x)
    softmax(a.H*tanh.(a.W*x.+a.b))
end

Flux.@functor Attention

"属性注意力机制"
struct AttrAttention
    encoders
    attention
end

""
function AttrAttention(in::Integer,out::Integer,as::Integer,init=Flux.glorot_uniform)
    AttrAttention(
        init(out,in,as),
        Attention(out,out)
    )
end

Flux.@functor AttrAttention

function (m::AttrAttention)(x)
    t,d,a,s,b=size(x)
    x1=reduce(
        vcat,
        map(eachslice(reshape(x,t*d,1,a,s*b),dims=4))do scale
            c=batched_mul(m.encoders,scale)
            e=Flux.flatten(c)
            sum(m.attention(e).*e,dims=2)
        end
    )
    reshape(x1,:,s,b)
end

struct ScaleAttention
    W
    b
    h
end

ScaleAttention(in::Integer;init=Flux.glorot_uniform)=ScaleAttention(init(in÷2,in),init(in÷2),init(1,in÷2))

function (m::ScaleAttention)(x)
    d,s,b=size(x)
    α=reduce(
        vcat,
        map(eachslice(x,dims=3))do t
            e=softmax(m.h*tanh.(m.W*t.+m.b),dims=2)
            vec(e)
        end
    )
    reshape(α,1,s,b).*x
end

Flux.@functor ScaleAttention

struct MSRD
    attratt
    scaleatt
    forward
    backward
    encoder
end

function MSRD(data::Dataset;out_features=512,hidden_features=out_features÷2)
    in_features=2*(data.ndiseases+data.ndrugs)
    as=prod(length ∘ values, values(data.similaritys))
    MSRD(
        AttrAttention(in_features,hidden_features,as),
        ScaleAttention(hidden_features),
        LSTM(hidden_features,out_features),
        LSTM(hidden_features,out_features),
        Dense(2out_features,out_features,sigmoid)
    )
end

Flux.@functor MSRD

function (m::MSRD)(x)
    x1=m.attratt(x)
    x2=m.scaleatt(x1)
    x3=reduce(
        hcat,
            map(eachslice(x2,dims=3))do t
            Flux.reset!(m.forward)
            for e in eachslice(t,dims=2)
                m.forward(e)
            end
            h,_=m.forward.state
            h
        end
    )
    x4=reduce(
        hcat,
        map(eachslice(x2,dims=3))do t
            Flux.reset!(m.backward)
            for e in eachslice(reverse(t,dims=2),dims=2)
                m.backward(e)
            end
            h,_=m.backward.state
            h
        end
    )
    x5=m.encoder(cat(x3,x4,dims=1))
end


struct CombineModel
    left
    right
    combine
end

function CombineModel(data::Dataset;feature_left=512,feature_right=512)
    CombineModel(
        MSRD(data;out_features=feature_left),
        RDConvolution(data;out=feature_right),
        Dense(feature_left+feature_right,2)
    )
end

function (m::CombineModel)(xₗ,xᵣ)
    yₗ=m.left(xₗ)
    yᵣ=m.right(xᵣ)
    y=m.combine(cat(yₗ,yᵣ,dims=1))
end


Flux.@functor CombineModel

export RDConvolution,Attention,AttrAttention,ScaleAttention,MSRD,CombineModel

