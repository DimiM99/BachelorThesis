from experiments.benchmark import run_benchmarks, print_results

fn main() raises:
    var n_runs = 5
    print("Starting benchmarks...")
    var results = run_benchmarks(n_runs)
    print_results(results, n_runs)
