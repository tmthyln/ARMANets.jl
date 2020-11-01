using Alert
using BSON: @save, @load
using Flux
using Images
using LinearAlgebra
using Metalhead
using ProgressMeter
using StatsBase

using ARMANets

function cifartrain(dataset, val_split = 0.01)
    training_set = trainimgs(dataset)
    first_img = first(training_set).img

    inputs = Array{Float32}(undef, size(first_img)..., 3, length(training_set))
    for i in 1:length(training_set)
        inputs[:, :, :, i] .= permutedims(channelview(training_set[i].img), (2, 3, 1))
    end

    ground_truth_classes = [training_set[i].ground_truth.class for i in 1:length(training_set)]
    outputs = Float32.(Flux.onehotbatch(ground_truth_classes, 1:10))

    val_examples = sample(1:length(training_set), Int(round(val_split * length(training_set))))
    train_examples = [i for i in 1:length(training_set) if i ∉ val_examples]

    train_data = (inputs[:, :, :, train_examples], outputs[:, train_examples])
    val_data = inputs[:, :, :, val_examples], outputs[:, val_examples]

    (Flux.Data.DataLoader(train_data, shuffle=true, batchsize=64), val_data)
end

function smalldata(dataset, num_examples = 32)
    training_set = trainimgs(dataset)
    first_img = first(training_set).img

    num_examples = min(num_examples, length(training_set))

    inputs = Array{Float32}(undef, size(first_img)..., 3, num_examples)
    for i in 1:num_examples
        inputs[:, :, :, i] .= permutedims(channelview(training_set[i].img), (2, 3, 1))
    end

    ground_truth_classes = [training_set[i].ground_truth.class for i in 1:num_examples]
    outputs = Float32.(Flux.onehotbatch(ground_truth_classes, 1:10))

    Flux.Data.DataLoader((inputs, outputs), shuffle=true, batchsize=8)
end

function conv_layer(kernel, dims, activation, arma = false)
    conv = Conv(kernel, dims, activation, pad=(1, 1), stride=(1, 1))
    if arma
        return Chain(conv, GeneralARMAConv(kernel, dims[2]))
    else
        return conv
    end
end

function network(use_arma = false)
    return Chain(
            conv_layer((3, 3), 3 => 64, relu, use_arma),
            conv_layer((3, 3), 64 => 64, relu, use_arma),
            MaxPool((2, 2)),
            conv_layer((3, 3), 64 => 128, relu, use_arma),
            conv_layer((3, 3), 128 => 128, relu, use_arma),
            MaxPool((2, 2)),
            conv_layer((3, 3), 128 => 256, relu, use_arma),
            conv_layer((3, 3), 256 => 256, relu, use_arma),
            MaxPool((2, 2)),
            conv_layer((3, 3), 256 => 512, relu, use_arma),
            conv_layer((3, 3), 512 => 512, relu, use_arma),
            conv_layer((3, 3), 512 => 512, relu, use_arma),
            MaxPool((2, 2)),
            flatten,
            Dense(2*2*512, 4096, relu),
            Dense(4096, 4096, relu),
            Dense(4096, 10))
end

function train_cifar!(device = cpu, max_epochs = 100)
    train, val = cifartrain(CIFAR10)
    x_val, y_val = val
    m = network() |> cpu
    loss(x, y) = Flux.logitcrossentropy(m(x), y)
    opt = ADAM(3e-4)

    local training_loss
    ps = params(m)
    for epoch in 1:max_epochs
        println("Epoch $epoch of $max_epochs")
        for (x, y) in train
            gs = gradient(ps) do
                training_loss = loss(device(x), device(y))
            end

            print("\tBatch Training Loss: $(training_loss)\t\t\r")
            Flux.update!(opt, ps, gs)
        end

        println("Validation Loss: $(loss(x_val, y_val))")
    end
    
    weights = params(cpu(m))
    @save "cifarmodel-checkpoint.bson" weights
    alert("Training complete.")

    return nothing
end
