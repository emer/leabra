# MPI Message Passing Interface Example

This is a version of the ra25 example that uses MPI to distributed computation across multiple processors (*procs*).  See [Wiki MPI](https://github.com/emer/emergent/wiki/MPI) for more info.

N completely separate instances of the same simulation program are run in parallel, and they communicate weight changes and trial-level log data amongst themselves.  Each proc thus trains on a subset of the total set of training patterns for each epoch.  Thus, dividing the patterns across procs is the most difficult aspect of making this work.  The mechanics of synchronizing the weight changes and etable data are just a few simple method calls.

Speedups approach linear, because the synchronization is relatively infrequent, especially for larger networks which have more computation per trial.  The biggest cost in MPI is the latency: sending a huge list of weight changes infrequently is much faster overall than sending smaller amounts of data more frequently.

You can only use MPI for running in nogui mode, using command-line args -- otherwise you'd get multiple copies of the GUI running..

# Building and running

To build with actual mpi support, you must do:

```bash
$ go build -tags mpi
```

otherwise it builds with a dummy version of mpi that doesn't actually do anything (convenient for enabling both MPI and non-MPI support in one codebase).  Always ensure that your code does something reasonable when mpi.WorldSize() == 1 -- that is what the dummy code returns  Also you should use a `UseMPI` flag, set by the `-mpi` command line arg, to do different things depending -- e.g., don't try to aggregate DWts if not using MPI, as it will waste a lot of time and accomplish nothing.

To run, do something like this:

```bash
$ mpirun -np 2 ./mpi -mpi
```

The number of processors must divide into 24 for this example (number of patterns used in ra25) evenly (2, 3, 4, 6, 8).

# General tips for MPI usage

* **MOST IMPORTANT:** all procs *must* remain *completely* synchronized in terms of when they call MPI functions -- these functions will block until all procs have called the same function.  The default behavior of setting a saved random number seed for all procs should ensure this.  But you also need to make sure that the same random permutation of item lists, etc takes place across all nodes.  The `empi.FixedTable` environment does this for the case of a table with a set of patterns.

* Instead of aggregating epoch-level stats directly on the Sim, which is how the basic ra25 example works, you need to record trial level data in an etable (`TrnTrlLog`), then synchronize that across all procs at the end of the epoch, and run aggregation stats on that data.  This is how the testing trial -> epoch process works in ra25 already.

# Key Diffs from ra25

Here are the main diffs that transform the ra25.go example into this mpi version:

* Search for `mpi` in the code (case insensitive) -- most of the changes have that in or near them.

* Most of the changes are the bottom of the file.

## main() Config() call

At the top of the file, it can be important to configure `TheSim` *after* mpi has been initialized, if there are things that are done differently there -- thus, you should move the `TheSim.Config()` call into `CmdArgs`:

```go
func main() {
	TheSim.New() // note: not running Config here -- done in CmdArgs for mpi / nogui
	if len(os.Args) > 1 {
		TheSim.CmdArgs() // simple assumption is that any args = no gui -- could add explicit arg if you want
	} else {
		TheSim.Config()      // for GUI case, config then run..
		gimain.Main(func() { // this starts gui -- requires valid OpenGL display connection (e.g., X11)
			guirun()
		})
	}
}
```

## Sim struct

There are some other things added but they are just more of what is already there -- these are the uniquely MPI parts, at end of Sim struct type:

```go
	UseMPI  bool      `view:"-" desc:"if true, use MPI to distribute computation across nodes"`
	Comm    *mpi.Comm `view:"-" desc:"mpi communicator"`
	AllDWts []float32 `view:"-" desc:"buffer of all dwt weight changes -- for mpi sharing"`
	SumDWts []float32 `view:"-" desc:"buffer of MPI summed dwt weight changes"`
```

## AlphaCyc

Now call the MPI version of WtFmDWt, which sums weight changes across procs:

```go
	if train {
		ss.MPIWtFmDWt() // special MPI version
	}
```

## Stats etc

Many changes in `InitStats` and `TrialStats`, removing manual aggregation.

## LogFileName

New version optionally adds the `rank` of the processor if not root -- sometimes it is useful for debugging or full stats to record a log for each processor, instead of just doing the root (0) one, which is the default.

```go
// LogFileName returns default log file name
func (ss *Sim) LogFileName(lognm string) string {
	nm := ss.Net.Nm + "_" + ss.RunName() + "_" + lognm
	if mpi.WorldRank() > 0 {
		nm += fmt.Sprintf("_%d", mpi.WorldRank())
	}
	nm += ".csv"
	return nm
}
```

## LogTrnEpc

Here we Gather the training trial-level data from all the procs, into a separate `TrnTrlLogAll` table, and use that if doing MPI:

```go
	trl := ss.TrnTrlLog
	if ss.UseMPI {
		empi.GatherTableRows(ss.TrnTrlLogAll, ss.TrnTrlLog, ss.Comm)
		trl = ss.TrnTrlLogAll
	}

	tix := etable.NewIdxView(trl)

	pcterr := agg.Mean(tix, "Err")[0]
```

The Gather appends all of the data from each proc in order by rank, in the resulting table.

There are also other changes in this method that remove the manual epoch-level computations, and just rely on the `agg.Mean` etc calls as illustrated above, and as used already in the testing code.

## CmdArgs

At the end, CmdArgs sets the `UseMPI` flag based on the `-mpi` arg, and has quite a bit of MPI-specific logic in it, which we don't reproduce here -- see `ra25.go` code and look for mpi.

We use `mpi.Printf` instead of `fmt.Printf` to have it only print on the root node, so you don't get a bunch of duplicated messages.

## MPI Code

The main MPI-specific code is at the end, reproduced here for easy reference.  NOTE: please always use the code in ra25.go as a copy-paste source as there might be a few small changes, which will be more closely tracked there than here.


```go
// MPIInit initializes MPI
func (ss *Sim) MPIInit() {
	mpi.Init()
	var err error
	ss.Comm, err = mpi.NewComm(nil) // use all procs
	if err != nil {
		log.Println(err)
		ss.UseMPI = false
	} else {
		mpi.Printf("MPI running on %d procs\n", mpi.WorldSize())
	}
}

// MPIFinalize finalizes MPI
func (ss *Sim) MPIFinalize() {
	if ss.UseMPI {
		mpi.Finalize()
	}
}

// CollectDWts collects the weight changes from all synapses into AllDWts
func (ss *Sim) CollectDWts(net *leabra.Network) {
    made := net.CollectDWts(&ss.AllDwts, 8329) // plug in number from printout below, to avoid realloc
    if made {
		mpi.Printf("MPI: AllDWts len: %d\n", len(ss.AllDWts)) // put this number in above make
	}
}

// MPIWtFmDWt updates weights from weight changes, using MPI to integrate
// DWt changes across parallel nodes, each of which are learning on different
// sequences of inputs.
func (ss *Sim) MPIWtFmDWt() {
	if ss.UseMPI {
		ss.CollectDWts(ss.Net)
		ndw := len(ss.AllDWts)
		if len(ss.SumDWts) != ndw {
			ss.SumDWts = make([]float32, ndw)
		}
		ss.Comm.AllReduceF32(mpi.OpSum, ss.SumDWts, ss.AllDWts)
		ss.Net.SetDWts(ss.SumDWts)
	}
	ss.Net.WtFmDWt()
}
```

