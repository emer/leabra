# MPI Message Passing Interface Example

This is a version of the ra25 example that uses MPI to distributed computation across multiple processors (*procs*).  See [Wiki MPI](https://github.com/emer/emergent/wiki/MPI) for more info.

N completely separate instances of the same simulation program are run in parallel, and they communicate weight changes and trial-level log data amongst themselves.  Each proc thus trains on a subset of the total set of training patterns for each epoch.  Thus, dividing the patterns across procs is the most difficult aspect of making this work.  The mechanics of synchronizing the weight changes and etable data are just a few simple method calls.

Speedups approach linear, because the synchronization is relatively infrequent, especially for larger networks which have more computation per trial.  The biggest cost in MPI is the latency: sending a huge list of weight changes infrequently is much faster overall than sending smaller amounts of data more frequently.

# Building and running

To build with actual mpi support, you must do:

```bash
$ go build -tags mpi
```

otherwise it builds with a dummy version of mpi that doesn't actually do anything (convenient for enabling both MPI and non-MPI support in one codebase).

To run, do something like this:

```bash
$ mpirun -np 2 ./mpi -mpi
```

The number of processors must divide into 24 (number of patterns) evenly (2, 3, 4, 6, 8).

# General tips for MPI usage

* **MOST IMPORTANT:** all procs *must* remain *completely* synchronized in terms of when they call MPI functions -- these functions will block until all procs have called the same function.  The default behavior of setting a saved random number seed for all procs should ensure this.  But you also need to make sure that the same random permutation of item lists, etc takes place across all nodes.  The `empi.FixedTable` environment does this for the case of a table with a set of patterns.

* Instead of aggregating epoch-level stats directly on the Sim, which is how the basic ra25 example works, you need to record trial level data in an etable (`TrnTrlLog`), then synchronize that across all procs at the end of the epoch, and run aggregation stats on that data.  This is how the testing trial -> epoch process works in ra25 already.



