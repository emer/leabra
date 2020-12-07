# bench

This is a standard benchmarking system for leabra.  It runs 5 layer fully connected networks of various sizes, with the number of events and epochs adjusted to take roughly an equal amount of time overall.

First, build the executable:

```sh
$ go build
```

* `run_bench.sh` is a script that runs standard configurations -- can pass additional args like `threads=2` to test different threading levels.

* `bench_results.md` has the algorithmic / implementational history for different versions of the code, on the same platform (macbook pro).

* `run_hardware.sh` is a script specifically for hardware testing, running standard 1, 2, 4 threads for each network size, and only reporting the final result, in the form shown in:

* `bench_hardware.md` has standard results for different hardware.


