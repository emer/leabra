# Reinforcement Learning and Dopamine

The `rl.go` file provides core infrastructure for dopamine neuromodulation and reinforcement learning, including the Rescorla-Wagner learning algorithm (RW) and Temporal Differences (TD) learning, and a minimal `ClampDaLayer` that can be used to send an arbitrary DA signal.

* `neuromod.go` has basic functions for sending neuromodulatory values such as DA.

* The RW and TD DA layers use the `SendMods` layer-level method to send the DA to other layers, at end of each cycle, after activation is updated.  Thus, DA lags by 1 cycle, which typically should not be a problem. 


