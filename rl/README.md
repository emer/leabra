# Reinforcement Learning and Dopamine

[![GoDoc](https://godoc.org/github.com/emer/leabra/rl?status.svg)](https://godoc.org/github.com/emer/leabra/rl)

The `rl` package provides core infrastructure for dopamine neuromodulation and reinforcement learning, including the Rescorla-Wagner learning algorithm (RW) and Temporal Differences (TD) learning, and a minimal `ClampDaLayer` that can be used to send an arbitrary DA signal.

* `da.go` defines a simple `DALayer` interface for getting and setting dopamine values, and a `SendDA` list of layer names that has convenience methods, and ability to send dopamine to any layer that implements the DALayer interface.

* The RW and TD DA layers use the `CyclePost` layer-level method to send the DA to other layers, at end of each cycle, after activation is updated.  Thus, DA lags by 1 cycle, which typically should not be a problem. 

* See the separate `pvlv` package for the full biologically based pvlv model on top of this basic DA infrastructure.


