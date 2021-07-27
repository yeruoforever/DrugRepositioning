function save(file,model)
    weights=params(cpu(model))
    BSON.@save file weights  
end

function load(file)
    BSON.@load file weights
    return weights
end