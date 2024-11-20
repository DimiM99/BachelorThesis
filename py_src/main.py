from experiments.benchmark import run_benchmarks, print_results

def main():
    n_runs = 5
    print("Starting benchmarks...")
    results = run_benchmarks(n_runs)
    print_results(results, n_runs)

if __name__ == "__main__":
    main()