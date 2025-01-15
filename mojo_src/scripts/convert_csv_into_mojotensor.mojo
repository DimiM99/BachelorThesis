from tensor import Tensor, TensorSpec, TensorShape
from pathlib import Path
from python import Python, PythonObject
from memory import UnsafePointer, memcpy
from algorithm import parallel_memcpy
from max.engine.tensor import EngineNumpyView
from utils.index import Index
from testing import assert_true

fn main() raises: 
    print("Preparing...")
    var np = Python.import_module("numpy")
    var pd = Python.import_module("pandas")
    print("Reading csv file via pandas")
    var data = pd.read_csv("./mnist.csv")
    print("Reading out the data shape")
    var rows = data.shape[0]
    var cols = data.shape[1]
    print("Preparing tensor shape, spec and emtpy tensor")
    var shape = TensorShape(rows, cols)
    var spec = TensorSpec(DType.float64, shape)
    var t_data = Tensor[DType.float64](spec)
    print("moving data to tensor")
    var np_arr = data.to_numpy()
    var np_arr_f = np.ascontiguousarray(np_arr, dtype='float64')
    var data_ptr = np_arr_f.__array_interface__['data'][0].unsafe_get_as_pointer[DType.float64]()
    memcpy(t_data.unsafe_ptr(), data_ptr, rows*cols)
    print("Data moved to tensor")
    t_data.save(Path("./mnist.mojotensor"))
    print("Data saved to file")
    print("Testing saved file")
    var t_data_loaded = Tensor[DType.float64].load(Path("./mnist.mojotensor"))
    print("Comparing..")
    assert_true(t_data.shape() == t_data_loaded.shape(), msg="Tensors did not match")
    print(t_data)
    print(t_data_loaded)
    print("Tensors matched")