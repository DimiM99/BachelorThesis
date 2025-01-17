from tensor import Tensor, TensorSpec, TensorShape
from pathlib import Path
from python import Python, PythonObject
from memory import UnsafePointer, memcpy
from algorithm import parallel_memcpy
from max.engine.tensor import EngineNumpyView
from utils.index import Index
from testing import assert_true

fn main() raises:
    alias type = DType.float64
    var shape = TensorShape(42000, 785)

    print("Testing saved file")
    var t_data_loaded = Tensor[type].load(Path("./mnist.mojotensor"))
    print("Comparing..")
    assert_true(shape == t_data_loaded.shape(), msg="Tensors did not match")
    print(t_data_loaded)
    print("Tensors matched")