from algorithm import vectorize
from sys import simdwidthof
from time import perf_counter_ns as now
from basalt import Graph, Symbol, OP, dtype, Tensor, TensorShape
from basalt.nn import Linear, ReLU, Softmax, CrossEntropyLoss, Model, optim
from basalt.utils.tensorutils import elwise_op, tmean, tstd, div
from basalt.autograd.attributes import AttributeVector, Attribute
from basalt.utils.dataloader import DataLoader

fn read_file(file_path: String) raises -> String:
    var s: String
    with open(file_path, "r") as f:
        s = f.read()
    return s


struct MNIST:
    var data: Tensor[dtype]
    var labels: Tensor[dtype]

    fn __init__(inout self, file_path: String) raises:
        var s = read_file(file_path)
        # Skip the first and last lines (featuresnames and last line is empty)
        var list_of_lines = s.split("\n")[1:-1]

        # Length is number of lines
        var N = len(list_of_lines)
        # Change shape to match pure Python version: N samples x 784 features
        self.data = Tensor[dtype](N, 784)
        self.labels = Tensor[dtype](N)

        var line: List[String] = List[String]()

        # Load data in Tensor
        for item in range(N):
            line = list_of_lines[item].split(",")
            self.labels[item] = atol(line[0])
            # Load directly into flattened 784 features
            for i in range(784):
                self.data[item * 784 + i] = atol(line[i + 1])

        # Normalize data
        alias nelts = simdwidthof[dtype]()

        @parameter
        fn vecdiv[nelts: Int](idx: Int):
            self.data.store[nelts](idx, div(self.data.load[nelts](idx), 255.0))

        vectorize[vecdiv, nelts](self.data.num_elements())

fn create_SimpleNN(batch_size: Int) -> Graph:
    var g = Graph()
    var x = g.input(TensorShape(batch_size, 784))

    var x_i = g.op(
        OP.TRANSPOSE, 
        x,
        attributes=AttributeVector(
            Attribute(
                "shape",
                TensorShape(x.shape[1], x.shape[0]),
            )
        )
    )

    ## Layers
    var z1 = Linear(g, x_i, 10)
    var a1 = ReLU(g, z1)
    var z2 = Linear(g, a1, 10)
    var a2 = Softmax(g, z2, axis=0)

    # Output
    g.out(a2)

    # Validation
    var y_true = g.input(TensorShape(batch_size, 10))
    var loss = CrossEntropyLoss(g, a2, y_true)
    g.loss(loss)

    return g^
      


fn prepare_data(
    mnist: MNIST,
    split_idx: Int = 1000
) raises -> (Tensor[dtype], Tensor[dtype], Tensor[dtype], Tensor[dtype]):
    var shape = mnist.data.shape()

    # Split into validation and training sets
    var X_dev = Tensor[dtype](split_idx, shape[1])  # First 1000 samples
    var Y_dev = Tensor[dtype](split_idx)
    var X_train = Tensor[dtype](shape[0] - split_idx, shape[1])  # Rest of samples
    var Y_train = Tensor[dtype](shape[0] - split_idx)

    # Copy validation data (first 1000 samples)
    for i in range(split_idx):
        Y_dev[i] = mnist.labels[i]
        for j in range(shape[1]):
            X_dev[i * shape[1] + j] = mnist.data[i * shape[1] + j]

    # Copy training data (rest of samples)
    for i in range(split_idx, shape[0]):
        Y_train[i - split_idx] = mnist.labels[i]
        for j in range(shape[1]):
            X_train[(i - split_idx) * shape[1] + j] = mnist.data[i * shape[1] + j]

    return X_train, Y_train, X_dev, Y_dev

fn main() raises:
    alias batch_size = 32
    alias num_epochs = 100
    alias learning_rate = 1e-3

    alias graph = create_SimpleNN(batch_size)

    var model = Model[graph]()
    var optim = optim.Adam[graph](model.parameters, lr=learning_rate)

    print("Loading data ...")
    # Load and prepare data
    var mnist = MNIST("./mnist.csv")
    X_train, Y_train, X_dev, Y_dev = prepare_data(mnist)

    var training_loader = DataLoader(
        data=X_train, 
        labels=Y_train, 
        batch_size=batch_size
    )

    var validation_loader = DataLoader(
        data=X_dev, 
        labels=Y_dev, 
        batch_size=batch_size
    )

    print("Training started...")
    var start = now()

    for epoch in range(num_epochs):
        var num_batches: Int = 0
        var epoch_loss: Float32 = 0.0
        var epoch_start = now()

        for batch in training_loader:
            # One-hot encode labels
            var labels_one_hot = Tensor[dtype](batch.labels.dim(0), 10)
            for b in range(batch.labels.dim(0)):
                labels_one_hot[int((b * 10 + batch.labels[b]))] = 1.0

            # Forward pass
            var loss = model.forward(batch.data, labels_one_hot)

            # Backward pass
            optim.zero_grad()
            model.backward()
            optim.step()

            epoch_loss += loss[0]
            num_batches += 1

            print(
                "Epoch [",
                epoch + 1,
                "/",
                num_epochs,
                "],\t Step [",
                num_batches,
                "/",
                X_train.dim(0) // batch_size,
                "],\t Loss:",
                epoch_loss / num_batches,
            )

        print("Epoch time: ", (now() - epoch_start) / 1e9, "seconds")

    print("Training finished: ", (now() - start) / 1e9, "seconds")
    model.print_perf_metrics("ms", True)

    print("\nEvaluating on validation set...")
    
    var val_loss: Float32 = 0.0
    var val_batches: Int = 0

    for batch in validation_loader:
        var val_labels_one_hot = Tensor[dtype](batch.labels.dim(0), 10)
        for b in range(batch.labels.dim(0)):
            val_labels_one_hot[int((b * 10 + batch.labels[b]))] = 1.0

        var val_output = model.forward(batch.data, val_labels_one_hot)
        val_loss += val_output[0]
        val_batches += 1

    print("Validation Loss:", val_loss / val_batches)
