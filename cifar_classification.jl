using Alert
using BSON: @save, @load
using Flux
using Images
using Metalhead
using StatsBase

using ARMANets

function cifartrain(dataset, val_split = 0.01)
    training_set = trainimgs(dataset)
    first_img = first(training_set).img

    inputs = Array{Float32}(undef, size(first_img)..., 3, length(training_set))
    for i in 1:length(training_set)
        inputs[:, :, :, i] = permutedims(channelview(training_set[i].img), (2, 3, 1))
    end

    ground_truth_classes = [training_set[i].ground_truth.class for i in 1:length(training_set)]
    outputs = Float32.(Flux.onehotbatch(ground_truth_classes, 1:10))

    val_examples = sample(1:length(training_set), Int(round(val_split * length(training_set))))
    train_examples = [i for i in 1:length(training_set) if i ∉ val_examples]

    train_data = (inputs[:, :, :, train_examples], outputs[:, train_examples])
    val_data = inputs[:, :, :, val_examples], outputs[:, val_examples]

    (Flux.Data.DataLoader(train_data, batchsize=8), val_data)
end

function network()
    return Chain(
            Conv((3, 3), 3 => 64, relu, pad=(1, 1), stride=(1, 1)),
            BatchNorm(64),
            Conv((3, 3), 64 => 64, relu, pad=(1, 1), stride=(1, 1)),
            BatchNorm(64),
            MaxPool((2,2)),
            Conv((3, 3), 64 => 128, relu, pad=(1, 1), stride=(1, 1)),
            BatchNorm(128),
            Conv((3, 3), 128 => 128, relu, pad=(1, 1), stride=(1, 1)),
            BatchNorm(128),
            MaxPool((2,2)),
            Conv((3, 3), 128 => 256, relu, pad=(1, 1), stride=(1, 1)),
            BatchNorm(256),
            Conv((3, 3), 256 => 256, relu, pad=(1, 1), stride=(1, 1)),
            BatchNorm(256),
            Conv((3, 3), 256 => 256, relu, pad=(1, 1), stride=(1, 1)),
            BatchNorm(256),
            Conv((3, 3), 256 => 256, relu, pad=(1, 1), stride=(1, 1)),
            MaxPool((2,2)),
            Conv((3, 3), 256 => 512, relu, pad=(1, 1), stride=(1, 1)),
            BatchNorm(512),
            Conv((3, 3), 512 => 512, relu, pad=(1, 1), stride=(1, 1)),
            BatchNorm(512),
            Conv((3, 3), 512 => 512, relu, pad=(1, 1), stride=(1, 1)),
            BatchNorm(512),
            Conv((3, 3), 512 => 512, relu, pad=(1, 1), stride=(1, 1)),
            MaxPool((2,2)),
            Conv((3, 3), 512 => 512, relu, pad=(1, 1), stride=(1, 1)),
            BatchNorm(512),
            Conv((3, 3), 512 => 512, relu, pad=(1, 1), stride=(1, 1)),
            BatchNorm(512),
            Conv((3, 3), 512 => 512, relu, pad=(1, 1), stride=(1, 1)),
            BatchNorm(512),
            Conv((3, 3), 512 => 512, relu, pad=(1, 1), stride=(1, 1)),
            MaxPool((2,2)),
            flatten,
            Dense(512, 4096, relu),
            Dropout(0.5),
            Dense(4096, 4096, relu),
            Dropout(0.5),
            Dense(4096, 10))
end

function train()
    train, val = cifartrain(CIFAR10)

    m = network()

    loss(x, y) = Flux.logitcrossentropy(m(x), y)

    evalcb = Flux.throttle(() -> (@info "Val Loss: $(loss(val...))"), 5)
    opt = ADAM(4e-5)

    for i in 1:5
        @info "Epoch $i:"
        Flux.train!(loss, params(m), train, opt, cb = evalcb)
    end

    @save "cifarmodel-checkpoint.bson" params(cpu(m))
    alert("Training complete.")
end
