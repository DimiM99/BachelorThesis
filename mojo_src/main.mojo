from experiments.benchmark_mojo import run_benchmarks, print_results
from experiments.bench_py_in_mojo import start_bench
from experiments.bench_t_snn import bench_t_snn
from experiments.NM_SNN import run_test

fn main() raises:
    # mojo benchmarks
    # var n_runs = 5
    # var results = run_benchmarks(n_runs)
    # print_results(results, n_runs)

    # Python benchmarks via mojo
    # start_bench()

    # bench tensor based SNN
    # bench_t_snn()

    # bench snn based on numojo 
    # run with ` mojo run -I "/Users/dimi/Developer/Code/OtherProjects/NuMojo" main.mojo` with magic shell aciive
    # run_test()
