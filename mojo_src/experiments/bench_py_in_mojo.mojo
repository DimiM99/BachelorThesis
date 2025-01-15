from python import Python, PythonObject

fn start_bench() raises:
    Python.add_to_path("../py_src")
    py_main_bench = Python.import_module("main")
    print("Starting Python benchmarks")
    py_main_bench.main()
    print("Python benchmarks finished")