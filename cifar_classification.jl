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

    (Flux.Data.DataLoader(train_data, shuffle=true, batchsize=32), val_data)
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

function network(conv = Conv)
    return Chain(
            conv((3, 3), 3 => 64, relu, pad=(1, 1), stride=(1, 1)),
            conv((3, 3), 64 => 64, relu, pad=(1, 1), stride=(1, 1)),
            MaxPool((2, 2)),
            conv((3, 3), 64 => 128, relu, pad=(1, 1), stride=(1, 1)),
            conv((3, 3), 128 => 128, relu, pad=(1, 1), stride=(1, 1)),
            MaxPool((2, 2)),
            conv((3, 3), 128 => 256, relu, pad=(1, 1), stride=(1, 1)),
            conv((3, 3), 256 => 256, relu, pad=(1, 1), stride=(1, 1)),
            MaxPool((2, 2)),
            conv((3, 3), 256 => 512, relu, pad=(1, 1), stride=(1, 1)),
            conv((3, 3), 512 => 512, relu, pad=(1, 1), stride=(1, 1)),
            conv((3, 3), 512 => 512, relu, pad=(1, 1), stride=(1, 1)),
            MaxPool((2, 2)),
            flatten,
            Dense(2*2*512, 4096, relu),
            Dense(4096, 4096, relu),
            Dense(4096, 10))
end

function train_cifar!()
    train = smalldata(CIFAR10)
    m = network() |> gpu
    loss(x, y) = Flux.logitcrossentropy(m(x), y)
    opt = ADAM(3e-4)

    local training_loss
    ps = params(m)
    for epoch in 1:20
        for (x, y) in train
            gs = gradient(ps) do
                training_loss = loss(gpu(x), gpu(y))
                return training_loss
            end

            println("\tTraining Loss on Batch: $(training_loss)")
            Flux.update!(opt, ps, gs)
            # Here you might like to check validation set accuracy
        end
    end
    
    weights = params(cpu(m))
    @save "cifarmodel-checkpoint.bson" weights
    alert("Training complete.")

    return nothing
end
