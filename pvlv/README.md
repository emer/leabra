# *WORK IN PROGRESS!*

# PVLV: Primary Value, Learned Value

[![GoDoc](https://godoc.org/github.com/emer/leabra/pvlv?status.svg)](https://godoc.org/github.com/emer/leabra/pvlv)


The PVLV model was ported from the model implemented in the C++ version of Emergent, and tries to faithfully reproduce the functionality of that model, as described in [Mollick et al, 2020](#references). This is the updated, bivalent version of PVLV, which has significant differences from the version created by the PVLV Wizard in C++ Emergent. Some portions of the code are very close to their C++ progenitors. At the same time, the Go version of the model follows the conventions of the new framework, so some aspects are quite different. The rewritten model's code should be on the whole more straightforward and easier to understand than the C++ version. That said, this is a very complex model, with a number of components that interact in ways that may not be immediately obvious.

## Example/Textbook Application

A detailed description of the PVLV model from the standpoint of psychological phenomena can be found in the [PVLV example application documentation](https://github.com/emer/leabra/blob/master/examples/pvlv/README.md). This document describes some parts of the implementation of that model.

## Overall architecture of the example application

The entry point for the whole program is the `main()` function, in `pvlv.go`. Just above `main()` in the code is a global variable, `TheSim`, which contains the top-level state of the entire simulation. Strictly speaking, it's not necessary to keep rogram state in a global variable, but since Go programs tend to have a great deal of concurrency, it greatly simplifies debugging. The very simple `main()` function collects any command-line parameters, sets their values in `TheSim`, and starts the graphical user interface ("GUI") for the program. One of the command-line parameters that can be supplied is `-nogui`, which will run the simulation with no user interface, for running experiments in batch mode (*Note*: batch mode is currently not fully implemented -- it will run the program, but needs more hooks to allow an experimental setup to be loaded).

### Command-line switches

See the [command line options](https://github.com/emer/leabra/blob/master/examples/pvlv/README.md#appendix-command-line-parameters) appendix of the PVLV textbook application documentation for a description of the available command-line switches.

The PVLV model uses the [`Stepper`](https://github.com/emer/emergent/blob/master/stepper/README.md) package to control running of the application.

## PVLV Library

### Network top level

PVLV extends `leabra.Network` to allow some extra 

### Layers

#### Inputs

PVLV has four types of inputs:
- Primary value, unconditioned inputs (US/PV): These represent stimuli that have inherent appetitive or aversive value, such as food, water, mating, electric shock, sharp edges, or nausea.
- Neutral-valued, conditioned inputs (CS/LV): These represent neutral stimuli such as bells or lights.
- Context inputs: These inputs are proxies for the environmental conditions associated with different CSs. These inputs provide a specific input, rather than absence of input, which is critical for extinction. In one dimension these inputs are in 1-to-1 correspondence with the set of CSs, while in the other dimension they correspond to time steps within a trial.
- USTime inputs: These inputs encode information about the upcoming occurrence of specific USs to ventral striatum patch compartment layers.

All inputs use localist representations.

#### Amygdala

The model has a total of 10 different amygdala layers, with three main divisions:
- Centrolateral amygdala (CEl): Two localist units for each US, one for acquisition, one for extinction. 
- Centromedial amygdala (CEm): One localist unit corresponding to each US. 
- Basolateral amygdala (BLA): Two groups of units corresponding to each US, one group for acquisition, one for extinction. This is the only portion of the model that uses distributed representations.

#### Ventral striatum
There are a total of 32 localist units representing the VS:
- 1 unit for each of 8 USs, times:
- D1 vs. D2 dopamine receptors, times:
- patch (striosome) vs. matrix compartments

#### Other layers
- Ventral Tegmental Area (VTA): Two single-unit nuclei, one each for appetitive and aversive stimuli.
- Lateral Habenula / Rostromedial Tegmental Nucleus (LHbRMTg): One unit, with inhibitory connections to both sides of the VTA.
- Pedunculopontine Tegmentum (PPTg): A single unit that relays a positively-rectified version of CEm output to the VTA

### Connectivity

- US inputs project directly to the acquisition portions of CEl, and to the appetitive D1-expressing and aversive D2-expressing portions of the BLA. US inputs are topographically ordered, with fixed weights.
- CS inputs project to these same portions of CEl and BLA, but with weak, diffuse connectivity that is highly plastic. In addition, CS inputs project to the matrix portion of VS, with weak connections whose strength is altered by learning.
- ContextIn units project primarily to portions of the model concerned with extinction.
- USTime inputs project only to the patch portions of VS.

### Modulation

For implementing modulation, sending layers implement `ModSender`, a simple interface with two methods, `SendMods` and `ModSendValue`. The implementation of `SendMods` calculates modulation values by summing subpool activation levels and storing the results in `ModSent`, then calling `ReceiveMods` on the layers in `ModReceivers`. Receivers implement the `ModReceiver` interface. `ReceiveMods` calls `ModSendValue` to get raw values, multiplies those by a scale factor stored in the `Scale` field of the receiver's `ModRcvrParams`, and stores the result in `ModNet`. `ModsFmInc`, the second method in the `ModReceiver` interface, is called later from the top-level network loop.

Modulatory projections do not convey activation directly, but act as a multiplier on activation from other excitatory inputs (with one exception, described below). In `ModsFmInc`, the value of `ModNet` is used to set `ModLevel` and `ModLrn` in the receiving layer. If `ModNet` is less than `ModNetThreshold`, `ModLrn` is set to 0, which completely shuts down weight changes on the receiver, while `ModLevel` is set to either 0 or 1 based on the `ActModZero` parameter on the receiving layer.

Activation in modulated layers is calculated by a customized version of `ActFmG`, which adjusts activation based on the value of `ModLevel`. The value of `nrn.Act` is copied into `ModAct`, which is altered based on DA activation. If a layer receives direct PV/US activation and PV activation is above threshold, `ModAct` is overridden by copying PV activation directly to `ModAct`. `ModAct` is used rather than `Act` in calculating weight changes.

`ModLevel` is implemented by a numeric value, but in the current implementation essentially acts as an all-or-nothing gate. A value of 1 in `ModLevel` allows full activation (i.e., no modulation), while a value of 0 completely shuts down activation on the receiving side.

Learning in PVLV is basically Hebbian. Weight changes are calculated in Q3, multiplying the activation level of the sending layer by the difference between the value of `ModAct` and the Q0 value of neuron activation, possibly suppressed by a zero value of `ModLrn`, whose calculation is described above (basically zero if `ModNet` is less than 0.1, otherwise 1). Learning is further modulated by the DA activation level. DA activation for learning is set to zero if it is less than the value of `DALrnThr` for the projection, further modified by `DALRBase` and `DALRGain`. Note that a zero value for DA activation will completely shut down learning, unless `DALRBase` is nonzero.

VS matrix layers use a delayed activity trace, similar to that described in the PBWM documentation. At each learning cycle, a raw value is calculated from the change in receiving neuron activation between Q3 qnd Q0, times sending layer activation, but rather than using this value directly, it is stored in a "trace", and the previous cycle's trace value determines weight changes. The model has provisions for keeping a decaying trace value and adding that into the value for the current cycle, but at present the decay function is essentially disabled, with a "decay" value set to 1.


# References

* Mollick, J. A., Hazy, T. E., Krueger, K. A., Nair, A., Mackie, P., Herd, S. A., & O’Reilly, R. C. (2020). A systems-neuroscience model of phasic dopamine. Psychological Review, Advance online publication. https://doi.org/10.1037/rev0000199
