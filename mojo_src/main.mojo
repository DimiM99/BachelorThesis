from experiments.benchmark_mojo import run_benchmarks, print_results
from experiments.bench_py_in_mojo import start_bench

fn main() raises:
    # mojo benchmarks
    var n_runs = 5
    var results = run_benchmarks(n_runs)
    print_results(results, n_runs)

    # Python benchmarks via mojo
    # start_bench()
