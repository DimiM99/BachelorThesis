from experiments.SNN_T import SimpleNN
from tensor import Tensor, TensorShape
from pathlib import Path
from algorithm import vectorize, parallelize
from sys.info import num_logical_cores
from time import perf_counter_ns

fn bench_t_snn() raises:
    alias type = DType.float64
    
    # Load data
    var t_data_loaded = Tensor[type].load(Path("mnist.mojotensor"))
    var shape = t_data_loaded.shape()
    alias split_idx = 1000
    
    # First split into validation and training sets
    var data_dev = extract_tensor_slice(
        t_data_loaded,
        Slice(0, split_idx),
        Slice(0, shape[1])
    )
    var data_train = extract_tensor_slice(
        t_data_loaded,
        Slice(split_idx, shape[0]),
        Slice(0, shape[1])
    )
    
    # Transpose both sets
    data_dev = SimpleNN._transpose_tensor(data_dev)
    data_train = SimpleNN._transpose_tensor(data_train)
    
    # Now extract features and labels from transposed data
    # For validation
    var Y_dev = extract_tensor_slice(
        data_dev,
        Slice(0, 1),  # First row now contains labels
        Slice(0, split_idx)
    )
    var X_dev = extract_tensor_slice(
        data_dev,
        Slice(1, shape[1]),  # Rest of the rows are features
        Slice(0, split_idx)
    )
    
    # For training
    var Y_train = extract_tensor_slice(
        data_train,
        Slice(0, 1),  # First row contains labels
        Slice(0, shape[0] - split_idx)
    )
    var X_train = extract_tensor_slice(
        data_train,
        Slice(1, shape[1]),  # Rest of the rows are features
        Slice(0, shape[0] - split_idx)
    )
    
    # Normalize features
    X_dev = X_dev / 255.0
    X_train = X_train / 255.0
    
    # Initialize and train the model
    var snn = SimpleNN(debug = True)
    var start = perf_counter_ns()
    snn.fit(X_train, Y_train, epochs = 500)
    var end = perf_counter_ns()

    # Evaluate
    var accuracy = snn.evaluate(X_dev, Y_dev)
    print("Validation accuracy:", accuracy)
    print("Training time:", (end - start) / 1e9, "seconds")

fn _transpose_tensor(x: Tensor[DType.float64]) -> Tensor[DType.float64]:
        var shape = x.shape()
        var result = Tensor[DType.float64](TensorShape(shape[1], shape[0]))
        if x.num_elements() < 768:
            for i in range(shape[0]):
                for j in range(shape[1]):
                    result[VariadicList(j, i)] = x[i, j]
        else:
            @parameter
            fn transpose_parallel(idx: Int):
                var i = idx // shape[1]
                var j = idx % shape[1]
                result[VariadicList(j, i)] = x[i, j]
            parallelize[transpose_parallel](shape[0] * shape[1], num_logical_cores())
        return result^

fn extract_tensor_slice(
    data: Tensor[DType.float64], 
    row_slice: Slice,
    col_slice: Slice
) -> Tensor[DType.float64]:
    var shape = data.shape()
    var num_rows = shape[0]
    var num_cols = shape[1]
    
    # Handle Optional[Int] fields properly
    var row_start = 0
    if row_slice.start:
        row_start = row_slice.start.value()
    
    var row_end = num_rows
    if row_slice.end:
        row_end = row_slice.end.value()
    
    var col_start = 0
    if col_slice.start:
        col_start = col_slice.start.value()
    
    var col_end = num_cols
    if col_slice.end:
        col_end = col_slice.end.value()
    #### 
    
    var result_rows = row_end - row_start
    var result_cols = col_end - col_start
    var result = Tensor[DType.float64](TensorShape(result_rows, result_cols))
    
    if result_rows * result_cols < 768:
        for i in range(row_start, row_end):
            for j in range(col_start, col_end):
                var src_idx = i * num_cols + j
                var dst_idx = (i - row_start) * result_cols + (j - col_start)
                result[dst_idx] = data[src_idx]
    else:
        @parameter
        fn copy_parallel(idx: Int):
            var i = idx // result_cols + row_start
            var j = idx % result_cols + col_start
            var src_idx = i * num_cols + j
            var dst_idx = (i - row_start) * result_cols + (j - col_start)
            result[dst_idx] = data[src_idx]
        parallelize[copy_parallel](result_rows * result_cols, num_logical_cores())
    
    return result^