# leabra in Go emergent

[![Go Report Card](https://goreportcard.com/badge/github.com/emer/leabra)](https://goreportcard.com/report/github.com/emer/leabra)
[![GoDoc](https://godoc.org/github.com/emer/leabra?status.svg)](https://godoc.org/github.com/emer/leabra)
[![Travis](https://travis-ci.com/emer/leabra.svg?branch=master)](https://travis-ci.com/emer/leabra)

This is the Go implementation of the Leabra algorithm for biologically-based models of cognition, based on the [Go emergent](https://github.com/emer/emergent) framework (with Python interface).

See [Wiki Install](https://github.com/emer/emergent/wiki/Install) for installation instructions, and the [Wiki Rationale](https://github.com/emer/emergent/wiki/Rationale) and [History](https://github.com/emer/emergent/wiki/History) pages for a more detailed rationale for the new version of emergent, and a history of emergent (and its predecessors).

# Current Status / News

* 11/2019: **nearing 1.0 complete release** -- [CCN textbook simulations](https://github.com/CompCogNeuro/sims) are (nearly) done and `hip`, `deep` and `pbwm` variants are in place and robustly tested.  Nearing full feature parity with C++ emergent!

* 6/12/2019: **Initial beta release** -- see the [ra25 example](https://github.com/emer/leabra/blob/master/examples/ra25) for a fully functional demo.  This example code is now sufficiently feature complete to provide a reasonable starting point for creating your own simulations, with both the Go and Python versions closely matched in functionality so you can choose either way (or explore both!).  We expect that although there will certainly be changes and improvements etc (there are many planned additions already), the existing code should be reasonably stable at this point, with only relatively minor changes to the API -- it is now safe to start building your own models!   Please file issues on github (use the emergent repository's issue tracker) for any bugs or other issues you find.

* 4/2019: separated this `leabra` repository from the overall `emergent` repository, to make it easier to fork and save / modify just the algorithm components of the system independent of the overall emergent infrastructure.  You just need to replace "github.com/emer/emergent/leabra/leabra" -> "github.com/emer/leabra/leabra" in your imports.

* 3/2019: Python interface is up and running!  See the `python` directory in `leabra` for the [README](https://github.com/emer/leabra/blob/master/python/README.md) status and how to give it a try.  You can run the full `examples/leabra25ra` code using Python, including the GUI etc.

# Design

* `leabra` sub-package provides a clean, well-organized implementation of core Leabra algorithms and Network structures. More specialized modifications such as `DeepLeabra` or `PBWM` or `PVLV` are all (going to be) implemented as additional specialized code that builds on / replaces elements of the basic version.  The goal is to make all of the code simpler, more transparent, and more easily modified by end users.  You should not have to dig through deep chains of C++ inheritance to find out what is going on.  Nevertheless, the basic tradeoffs of code re-use dictate that not everything should be in-line in one massive blob of code, so there is still some inevitable tracking down of function calls etc.  The algorithm overview below should be helpful in finding everything.

* `ActParams` (in [act.go](https://github.com/emer/leabra/blob/master/leabra/act.go)), `InhibParams` (in [inhib.go](https://github.com/emer/leabra/blob/master/leabra/inhib.go)), and `LearnNeurParams` / `LearnSynParams` (in [learn.go](https://github.com/emer/leabra/blob/master/leabra/learn.go)) provide the core parameters and functions used, including the X-over-X-plus-1 activation function, FFFB inhibition, and the XCal BCM-like learning rule, etc.  This function-based organization should be clearer than the purely structural organization used in C++ emergent.

* There are 3 main levels of structure: `Network`, `Layer` and `Prjn` (projection).  The network calls methods on its Layers, and Layers iterate over both `Neuron` data structures (which have only a minimal set of methods) and the `Prjn`s, to implement the relevant computations.  The `Prjn` fully manages everything about a projection of connectivity between two layers, including the full list of `Syanpse` elements in the connection.  There is no "ConGroup" or "ConState" level as was used in C++, which greatly simplifies many things.  The Layer also has a set of `Pool` elements, one for each level at which inhibition is computed (there is always one for the Layer, and then optionally one for each Sub-Pool of units (*Pool* is the new simpler term for "Unit Group" from C++ emergent).

* The `NetworkStru` and `LayerStru` structs manage all the core structural aspects of things (data structures etc), and then the algorithm-specific versions (e.g., `leabra.Network`) use Go's anonymous embedding (akin to inheritance in C++) to transparently get all that functionality, while then directly implementing the algorithm code.  Almost every step of computation has an associated method in `leabra.Layer`, so look first in [layer.go](https://github.com/emer/leabra/blob/master/leabra/layer.go) to see how something is implemented.  

* Each structural element directly has all the parameters controlling its behavior -- e.g., the `Layer` contains an `ActParams` field (named `Act`), etc, instead of using a separate `Spec` structure as in C++ emergent.  The Spec-like ability to share parameter settings across multiple layers etc is instead achieved through a **styling**-based paradigm -- you apply parameter "styles" to relevant layers instead of assigning different specs to them.  This paradigm should be less confusing and less likely to result in accidental or poorly-understood parameter applications.  We adopt the CSS (cascading-style-sheets) standard where parameters can be specifed in terms of the Name of an object (e.g., `#Hidden`), the *Class* of an object (e.g., `.TopDown` -- where the class name TopDown is manually assigned to relevant elements), and the *Type* of an object (e.g., `Layer` applies to all layers).  Multiple space-separated classes can be assigned to any given element, enabling a powerful combinatorial styling strategy to be used.

* Go uses `interfaces` to represent abstract collections of functionality (i.e., sets of methods).  The `emer` package provides a set of interfaces for each structural level (e.g., `emer.Layer` etc) -- any given specific layer must implement all of these methods, and the structural containers (e.g., the list of layers in a network) are lists of these interfaces.  An interface is implicitly a *pointer* to an actual concrete object that implements the interface.  Thus, we typically need to convert this interface into the pointer to the actual concrete type, as in:

```Go
func (nt *Network) InitActs() {
	for _, ly := range nt.Layers {
		if ly.IsOff() {
			continue
		}
		ly.(*Layer).InitActs() // ly is the emer.Layer interface -- (*Layer) converts to leabra.Layer
	}
}
```

* The emer interfaces are designed to support generic access to network state, e.g., for the 3D network viewer, but specifically avoid anything algorithmic.  Thus, they should allow viewing of any kind of network, including PyTorch backprop nets.

* There is also a `leabra.LeabraLayer` and `leabra.LeabraPrjn` interface, defined in [leabra.go](https://github.com/emer/leabra/blob/master/leabra/leabra.go), which provides a virtual interface for the Leabra-specific algorithm functions at the basic level.  These interfaces are used in the base leabra code, so that any more specialized version that embeds the basic leabra types will be called instead.  See `deep` sub-package for implemented example that does DeepLeabra on top of the basic `leabra` foundation.

* Layers have a `Shape` property, using the `etensor.Shape` type, which specifies their n-dimensional (tensor) shape.  Standard layers are expected to use a 2D Y*X shape (note: dimension order is now outer-to-inner or *RowMajor* now), and a 4D shape then enables `Pools` ("unit groups") as hypercolumn-like structures within a layer that can have their own local level of inihbition, and are also used extensively for organizing patterns of connectivity.

# Naming Conventions

There are several changes from the original C++ emergent implementation for how things are named now:
* `Pool <- Unit_Group` -- A group of Neurons that share pooled inhibition.  Can be entire layer and / or sub-pools within a layer.
* `AlphaCyc <- Trial` -- We are now distinguishing more clearly between network-level timing units (e.g., the 100 msec alpha cycle over which learning operates within posterior cortex) and environmental or experimental timing units, e.g., the `Trial` etc. Please see the [TimeScales](https://godoc.org/github.com/emer/leabra/leabra#TimeScales) type for an attempt to standardize the different units of time along these different dimensions.  The `examples/ra25` example uses trials and epochs for controlling the "environment" (such as it is), while the algorithm-specific code refers to AlphaCyc, Quarter, and Cycle, which are the only time scales that are specifically coded within the algorithm -- everything else is up to the specific model code.

# The Leabra Algorithm

Leabra stands for *Local, Error-driven and Associative, Biologically Realistic Algorithm*, and it implements a balance between error-driven (backpropagation) and associative (Hebbian) learning on top of a biologically-based point-neuron activation function with inhibitory competition dynamics (either via inhibitory interneurons or an approximation thereof), which produce k-Winners-Take-All (kWTA) sparse distributed representations.  Extensive documentation is available from the online textbook: [Computational Cognitive Neuroscience](http://ccnbook.colorado.edu) which serves as a second edition to the original book: *Computational Explorations in Cognitive Neuroscience: Understanding
the Mind by Simulating the Brain*, O'Reilly and Munakata, 2000,
Cambridge, MA: MIT Press. [Computational Explorations..]([https://psych.colorado.edu/~oreilly/comp_ex_cog_neuro.html)

The name is pronounced like "Libra" and is intended to connote the *balance* of various different factors in an attempt to approach the "golden middle" ground between biological realism and computational efficiency and the ability to simulate complex cognitive function.

The version of Leabra implemented here corresponds to version 8.5 of [C++ emergent](https://grey.colorado.edu/emergent/index.php/Main_Page).

The basic activation dynamics of Leabra are based on standard electrophysiological principles of real neurons, and in discrete spiking mode we implement exactly the AdEx (adapting exponential) model of Gerstner and colleagues [Scholarpedia article on AdEx](https://www.scholarpedia.org/article/Adaptive_exponential_integrate-and-fire_model).  The basic `leabra` package implements the rate code mode (which runs faster and allows for smaller networks), which provides a very close approximation to the AdEx model behavior, in terms of a graded activation signal matching the actual instantaneous rate of spiking across a population of AdEx neurons.  We generally conceive of a single rate-code neuron as representing a microcolumn of roughly 100 spiking pyramidal neurons in the neocortex.

The excitatory synaptic input conductance (`Ge` in the code, known as *net input* in artificial neural networks) is computed as an average, not a sum, over connections, based on normalized, sigmoidaly transformed weight values, which are subject to scaling on a connection-group level to alter relative contributions.  Automatic scaling is performed to
compensate for differences in expected activity level in the different projections.  See section on [Input Scaling](#input-scaling) for details.

Inhibition is computed using a feed-forward (FF) and feed-back (FB) inhibition function (*FFFB*) that closely approximates the behavior of inhibitory interneurons in the neocortex.  FF is based on a multiplicative factor applied to the average excitatory net input coming into a layer, and FB is based on a multiplicative factor applied to the average activation within the layer.  These simple linear functions do an excellent job of controlling the overall activation levels in bidirectionally connected networks, producing behavior very similar to the more abstract computational implementation of kWTA dynamics implemented in previous versions.

There is a single learning equation, derived from a very detailed model of spike timing dependent plasticity (STDP) by Urakubo, Honda, Froemke, et al (2008), that produces a combination of Hebbian associative and error-driven learning.  For historical reasons, we call this the *XCAL* equation (*eXtended Contrastive Attractor Learning*), and it is functionally very similar to the *BCM* learning rule developed by Bienenstock, Cooper, and Munro (1982).  The essential learning dynamic involves a Hebbian-like co-product of sending neuron activation times receiving neuron activation, which biologically reflects the amount of calcium entering through NMDA channels, and this co-product is then compared against a floating threshold value.  To produce the Hebbian learning dynamic, this floating threshold is based on a longer-term running average of the receiving neuron activation (`AvgL` in the code).  This is the key idea for the BCM algorithm.  To produce error-driven learning, the floating threshold is based on a faster running average of activation co-products (`AvgM`), which reflects an expectation or prediction, against which the instantaneous, later outcome is compared.

Weights are subject to a contrast enhancement function, which compensates for the soft (exponential) weight bounding that keeps weights within the normalized 0-1 range.  Contrast enhancement is important for enhancing the selectivity of self-organizing learning, and generally results in faster learning with better overall results.  Learning operates on the underlying internal linear weight value.  Biologically, we associate the underlying linear weight value with internal synaptic factors such as actin scaffolding, CaMKII phosphorlation level, etc, while the contrast enhancement operates at the level of AMPA receptor expression.

There are various extensions to the algorithm that implement special neural mechanisms associated with the prefrontal cortex and basal ganglia [PBWM](#pbwm), dopamine systems [PVLV](#pvlv), the [Hippocampus](#hippocampus), and predictive learning and temporal integration dynamics associated with the thalamocortical circuits [DeepLeabra](#deepleabra).  All of these are (will be) implemented as additional modifications of the core, simple `leabra` implementation, instead of having everything rolled into one giant hairball as in the original C++ implementation.

# Pseudocode as a LaTeX doc for Paper Appendix

You can copy the mediawiki source of the following section into a file, and run [pandoc](https://pandoc.org/) on it to convert to LaTeX (or other formats) for inclusion in a paper.  As this wiki page is always kept updated, it is best to regenerate from this source -- very easy:

```bash
wget "https://grey.colorado.edu/emergent/index.php/Leabra?action=raw" -o appendix.mw
pandoc appendix.mw -f mediawiki -t latex -o appendix.tex
</pre>
The \href to LeabraNetinScaling will have to be manually edited, and there is a \tightlist macro that is defined as:
<pre>
\providecommand{\tightlist}{%
  \setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
```

which you need to include in the header of your .tex file.


# Leabra Algorithm Equations

The pseudocode for Leabra is given here, showing exactly how the pieces of the algorithm fit together, using the equations and variables from the actual code.  Compared to the original C++ emergent implementation, this Go version of emergent is much more readable, while also not being too much slower overall.

There are also other implementations of Leabra available:
* [leabra7](https://github.com/PrincetonUniversity/leabra7) Python implementation of the version 7 of Leabra, by Daniel Greenidge and Ken Norman at Princeton.
* [Matlab](https://grey.colorado.edu/cgi-bin/viewvc.cgi/emergent/trunk/Matlab/) (also available in the C++ emergent source tree) -- a complete implementation of these equations in Matlab, coded by Sergio Verduzco-Flores.
* [Python](https://github.com/benureau/leabra) implementation by Fabien Benureau.
* [R](https://github.com/johannes-titz/leabRa) implementation by Johannes Titz.

This repository contains specialized additions to the core algorithm described here:
* [deep](https://github.com/emer/leabra/blob/master/deep) has the DeepLeabra mechanisms for simulating the deep neocortical <-> thalamus pathways (wherein basic Leabra represents purely superficial-layer processing)
* [pbwm](https://github.com/emer/leabra/blob/master/pbwm) has the prefrontal-cortex basal ganglia working memory model (PBWM) and associated learning mechanisms such as TD (temporal differences).
* [hip](https://github.com/emer/leabra/blob/master/hip) has the hippocampus specific learning mechanisms.

## Timing

Leabra is organized around the following timing, based on an internally-generated alpha-frequency (10 Hz, 100 msec periods) cycle of expectation followed by outcome, supported by neocortical circuitry in the deep layers and the thalamus, as hypothesized in the [DeepLeabra](#deepleabra) extension to standard Leabra:

* A **Trial** lasts 100 msec (10 Hz, alpha frequency), and comprises one sequence of expectation -- outcome learning, organized into 4 quarters.
    + Biologically, the deep neocortical layers (layers 5, 6) and the thalamus have a natural oscillatory rhythm at the alpha frequency.  Specific dynamics in these layers organize the cycle of expectation vs. outcome within the alpha cycle.
    
* A **Quarter** lasts 25 msec (40 Hz, gamma frequency) -- the first 3 quarters (75 msec) form the expectation / minus phase, and the final quarter are the outcome / plus phase.
    + Biologically, the superficial neocortical layers (layers 2, 3) have a gamma frequency oscillation, supporting the quarter-level organization.
    
* A **Cycle** represents 1 msec of processing, where each neuron updates its membrane potential etc according to the above equations.

## Variables

The `leabra.Neuron` struct contains all the neuron (unit) level variables, and the `leabra.Layer` contains a simple Go slice of these variables.  Optionally, there can be `leabra.Pool` pools of subsets of neurons that correspond to hypercolumns, and support more local inhibitory dynamics (these used to be called UnitGroups in the C++ version).

* `Act`   = overall rate coded activation value -- what is sent to other neurons -- typically in range 0-1
* `Ge` = total excitatory synaptic conductance -- the net excitatory input to the neuron -- does *not* include Gbar.E
* `Gi` = total inhibitory synaptic conductance -- the net inhibitory input to the neuron -- does *not* include Gbar.I
* `Inet` = net current produced by all channels -- drives update of Vm
* `Vm` = membrane potential -- integrates Inet current over time
* `Targ` = target value: drives learning to produce this activation value
* `Ext` = external input: drives activation of unit from outside influences (e.g., sensory input)
* `AvgSS` = super-short time-scale activation average -- provides the lowest-level time integration -- for spiking this integrates over spikes before subsequent averaging, and it is also useful for rate-code to provide a longer time integral overall
* `AvgS` = short time-scale activation average -- tracks the most recent activation states (integrates over AvgSS values), and represents the plus phase for learning in XCAL algorithms
* `AvgM` = medium time-scale activation average -- integrates over AvgS values, and represents the minus phase for learning in XCAL algorithms
* `AvgL` = long time-scale average of medium-time scale (trial level) activation, used for the BCM-style floating threshold in XCAL
* `AvgLLrn` = how much to learn based on the long-term floating threshold (AvgL) for BCM-style Hebbian learning -- is modulated by level of AvgL itself (stronger Hebbian as average activation goes higher) and optionally the average amount of error experienced in the layer (to retain a common proportionality with the level of error-driven learning across layers)
* `AvgSLrn` = short time-scale activation average that is actually used for learning -- typically includes a small contribution from AvgM in addition to mostly AvgS, as determined by `LrnActAvgParams.LrnM` -- important to ensure that when unit turns off in plus phase (short time scale), enough medium-phase trace remains so that learning signal doesn't just go all the way to 0, at which point no learning would take place
* `ActM` = records the traditional posterior-cortical minus phase activation, as activation after third quarter of current alpha cycle
* `ActP` = records the traditional posterior-cortical plus_phase activation, as activation at end of current alpha cycle
* `ActDif` = ActP - ActM -- difference between plus and minus phase acts -- reflects the individual error gradient for this neuron in standard error-driven learning terms
* `ActDel` delta activation: change in Act from one cycle to next -- can be useful to track where changes are taking place
* `ActAvg` = average activation (of final plus phase activation state) over long time intervals (time constant = DtPars.AvgTau -- typically 200) -- useful for finding hog units and seeing overall distribution of activation
* `Noise` = noise value added to unit (`ActNoiseParams` determines distribution, and when / where it is added)
* `GiSyn` = aggregated synaptic inhibition (from Inhib projections) -- time integral of GiRaw -- this is added with computed FFFB inhibition to get the full inhibition in Gi
* `GiSelf` = total amount of self-inhibition -- time-integrated to avoid oscillations

The following are more implementation-level variables used in integrating synaptic inputs:

* `ActSent` = last activation value sent (only send when diff is over threshold)
* `GeRaw` = raw excitatory conductance (net input) received from sending units (send delta's are added to this value)
* `GeInc` = delta increment in GeRaw sent using SendGeDelta
* `GiRaw` = raw inhibitory conductance (net input) received from sending units (send delta's are added to this value)
* `GiInc` = delta increment in GiRaw sent using SendGeDelta

Neurons are connected via synapses parameterized with the following variables, contained in the `leabra.Synapse` struct.  The `leabra.Prjn` contains all of the synaptic connections for all the neurons across a given layer -- there are no Neuron-level data structures in the Go version.  

* `Wt` = synaptic weight value -- sigmoid contrast-enhanced
* `LWt` = linear (underlying) weight value -- learns according to the lrate specified in the connection spec -- this is converted into the effective weight value, Wt, via sigmoidal contrast enhancement (see `WtSigParams`)
* `DWt` = change in synaptic weight, from learning
* `Norm` = DWt normalization factor -- reset to max of abs value of DWt, decays slowly down over time -- serves as an estimate of variance in weight changes over time
* `Moment` = momentum -- time-integrated DWt changes, to accumulate a consistent direction of weight change and cancel out dithering contradictory changes

## Activation Update Cycle (every 1 msec): Ge, Gi, Act

The `leabra.Network` `Cycle` method in `leabra/network.go` looks like this:

```Go
// Cycle runs one cycle of activation updating:
// * Sends Ge increments from sending to receiving layers
// * Average and Max Ge stats
// * Inhibition based on Ge stats and Act Stats (computed at end of Cycle)
// * Activation from Ge, Gi, and Gl
// * Average and Max Act stats
// This basic version doesn't use the time info, but more specialized types do, and we
// want to keep a consistent API for end-user code.
func (nt *Network) Cycle(ltime *Time) {
	nt.SendGDelta(ltime) // also does integ
	nt.AvgMaxGe(ltime)
	nt.InhibFmGeAct(ltime)
	nt.ActFmG(ltime)
	nt.AvgMaxAct(ltime)
}
```

For every cycle of activation updating, we compute the excitatory input conductance `Ge`, then compute inhibition `Gi` based on average `Ge` and `Act` (from previous cycle), then compute the `Act` based on those conductances.  The equations below are not shown in computational order but rather conceptual order for greater clarity.  All of the relevant parameters are in the `leabra.Layer.Act` and `Inhib` fields, which are of type `ActParams` and `InhibParams` -- in this Go version, the parameters have been organized functionally, not structurally, into three categories.

* `Ge` excitatory conductance is actually computed using a highly efficient delta-sender-activation based algorithm, which only does the expensive multiplication of activations * weights when the sending activation changes by a given amount (`OptThreshParams.Delta`).  However, conceptually, the conductance is given by this equation:
    + `GeRaw += Sum_(recv) Prjn.GScale * Send.Act * Wt`
        + `Prjn.GScale` is the [Input Scaling](#input-scaling) factor that includes 1/N to compute an average, and the `WtScaleParams` `Abs` absolute scaling and `Rel` relative scaling, which allow one to easily modulate the overall strength of different input projections.
    + `Ge += DtParams.Integ * (1/ DtParams.GTau) * (GeRaw - Ge)`
        + This does a time integration of excitatory conductance, `GTau = 1.4` default, and global integration time constant, `Integ = 1` for 1 msec default.

* `Gi` inhibtory conductance combines computed and synaptic-level inhibition (if present) -- most of code is in `leabra/inhib.go`
    + `ffNetin = avgGe + FFFBParams.MaxVsAvg * (maxGe - avgGe)`
    + `ffi = FFFBParams.FF * MAX(ffNetin - FFBParams.FF0, 0)`
        + feedforward component of inhibition with FF multiplier (1 by default) -- has FF0 offset and can't be negative (that's what the MAX(.. ,0) part does).
        + `avgGe` is average of Ge variable across relevant Pool of neurons, depending on what level this is being computed at, and `maxGe` is max of Ge across Pool
    + `fbi += (1 / FFFBParams.FBTau) * (FFFBParams.FB * avgAct - fbi`
        + feedback component of inhibition with FB multiplier (1 by default) -- requires time integration to dampen oscillations that otherwise occur -- FBTau = 1.4 default.
    + `Gi = FFFBParams.Gi * (ffi + fbi)`
        + total inhibitory conductance, with global Gi multiplier -- default of 1.8 typically produces good sparse distributed representations in reasonably large layers (25 units or more).

* `Act` activation from Ge, Gi, Gl (most of code is in `leabra/act.go`, e.g., `ActParams.ActFmG` method).  When neurons are above thresholds in subsequent condition, they obey the "geLin" function which is linear in Ge:
    + `geThr = (Gi * (Erev.I - Thr) + Gbar.L * (Erev.L - Thr) / (Thr - Erev.E)`
    + `nwAct = NoisyXX1(Ge * Gbar.E - geThr)`
        + geThr = amount of excitatory conductance required to put the neuron exactly at the firing threshold, `XX1Params.Thr` = .5 default, and NoisyXX1 is the x / (x+1) function convolved with gaussian noise kernel, where x = `XX1Parms.Gain` * Ge - geThr) and Gain is 100 by default
    + `if Act < XX1Params.VmActThr && Vm <= X11Params.Thr: nwAct = NoisyXX1(Vm - Thr)`
        + it is important that the time to first "spike" (above-threshold activation) be governed by membrane potential Vm integration dynamics, but after that point, it is essential that activation drive directly from the excitatory conductance Ge relative to the geThr threshold.
    + `Act += (1 / DTParams.VmTau) * (nwAct - Act)`
        + time-integration of the activation, using same time constant as Vm integration (VmTau = 3.3 default)
    + `Vm += (1 / DTParams.VmTau) * Inet`
    + `Inet = Ge * (Erev.E - Vm) + Gbar.L * (Erev.L - Vm) + Gi * (Erev.I - Vm) + Noise`
        + Membrane potential computed from net current via standard RC model of membrane potential integration.  In practice we use normalized Erev reversal potentials and Gbar max conductances, derived from biophysical values: Erev.E = 1, .L = 0.3, .I = 0.25, Gbar's are all 1 except Gbar.L = .2 default.

## Learning

![XCAL DWt Function](fig_xcal_dwt_fun.png?raw=true "The XCAL dWt function, showing direction and magnitude of synaptic weight changes dWt as a function of the short-term average activity of the sending neuron *x* times the receiving neuron *y*.  This quantity is a simple mathematical approximation to the level of postsynaptic Ca++, reflecting the dependence of the NMDA channel on both sending and receiving neural activity.  This function was extracted directly from the detailed biophysical Urakubo et al. 2008 model, by fitting a piecewise linear function to the synaptic weight change behavior that emerges from it as a function of a wide range of sending and receiving spiking patterns.")

Learning is based on running-averages of activation variables, parameterized in the `leabra.Layer.Learn` `LearnParams` field, mostly implemented in the `leabra/learn.go` file.

* **Running averages** computed continuously every cycle, and note the compounding form.  Tau params in `LrnActAvgParams`:
    + `AvgSS += (1 / SSTau) * (Act - AvgSS)`
        + super-short time scale running average, SSTau = 2 default -- this was introduced to smooth out discrete spiking signal, but is also useful for rate code.
    + `AvgS += (1 / STau) * (AvgSS - AvgS)`
        + short time scale running average, STau = 2 default -- this represents the *plus phase* or actual outcome signal in comparison to AvgM
    + `AvgM += (1 / MTau) * (AvgS - AvgM)`
        + medium time-scale running average, MTau = 10 -- this represents the *minus phase* or expectation signal in comparison to AvgS
    + `AvgL += (1 / Tau) * (Gain * AvgM - AvgL); AvgL = MAX(AvgL, Min)`
        + long-term running average -- this is computed just once per learning trial, *not every cycle* like the ones above -- params on `AvgLParams`: Tau = 10, Gain = 2.5 (this is a key param -- best value can be lower or higher) Min = .2
    + `AvgLLrn = ((Max - Min) / (Gain - Min)) * (AvgL - Min)`
        + learning strength factor for how much to learn based on AvgL floating threshold -- this is dynamically modulated by strength of AvgL itself, and this turns out to be critical -- the amount of this learning increases as units are more consistently active all the time (i.e., "hog" units).  Params on `AvgLParams`, Min = 0.0001, Max = 0.5. Note that this depends on having a clear max to AvgL, which is an advantage of the exponential running-average form above.
    + `AvgLLrn *= MAX(1 - layCosDiffAvg, ModMin)`
        + also modulate by time-averaged cosine (normalized dot product) between minus and plus phase activation states in given receiving layer (layCosDiffAvg), (time constant 100) -- if error signals are small in a given layer, then Hebbian learning should also be relatively weak so that it doesn't overpower it -- and conversely, layers with higher levels of error signals can handle (and benefit from) more Hebbian learning.  The MAX(ModMin) (ModMin = .01) factor ensures that there is a minimum level of .01 Hebbian (multiplying the previously-computed factor above).  The .01 * .05 factors give an upper-level value of .0005 to use for a fixed constant AvgLLrn value -- just slightly less than this (.0004) seems to work best if not using these adaptive factors.
    + `AvgSLrn = (1-LrnM) * AvgS + LrnM * AvgM`
        + mix in some of the medium-term factor into the short-term factor -- this is important for ensuring that when neuron turns off in the plus phase (short term), that enough trace of earlier minus-phase activation remains to drive it into the LTD weight decrease region -- LrnM = .1 default.

* **Learning equation**:
    + `srs = Send.AvgSLrn * Recv.AvgSLrn`
    + `srm = Send.AvgM * Recv.AvgM`
    + `dwt = XCAL(srs, srm) + Recv.AvgLLrn * XCAL(srs, Recv.AvgL)`
        + weight change is sum of two factors: error-driven based on medium-term threshold (srm), and BCM Hebbian based on long-term threshold of the recv unit (Recv.AvgL)
    + XCAL is the "check mark" linearized BCM-style learning function (see figure) that was derived from the Urakubo Et Al (2008) STDP model, as described in more detail in the [CCN textbook](http://ccnbook.colorado.edu)
        + `XCAL(x, th) = (x < DThr) ? 0 : (x > th * DRev) ? (x - th) : (-x * ((1-DRev)/DRev))`
        + DThr = 0.0001, DRev = 0.1 defaults, and x ? y : z terminology is C syntax for: if x is true, then y, else z

    + **DWtNorm** -- normalizing the DWt weight changes is standard in current backprop, using the AdamMax version of the original RMS normalization idea, and benefits Leabra as well, and is On by default, params on `DwtNormParams`:
        + `Norm = MAX((1 - (1 / DecayTau)) * Norm, ABS(dwt))`
            + increment the Norm normalization using abs (L1 norm) instead of squaring (L2 norm), and with a small amount of decay: DecayTau = 1000.
        + `dwt *= LrComp / MAX(Norm, NormMin)`
            + normalize dwt weight change by the normalization factor, but with a minimum to prevent dividing by 0 -- LrComp compensates overall learning rate for this normalization (.15 default) so a consistent learning rate can be used, and NormMin = .001 default.
    + **Momentum** -- momentum is turned On by default, and has significant benefits for preventing hog units by driving more rapid specialization and convergence on promising error gradients.  Parameters on `MomentumParams`:
        + `Moment = (1 - (1 / MTau)) * Moment + dwt`
        + `dwt = LrComp * Moment`
            + increment momentum from new weight change, MTau = 10, corresponding to standard .9 momentum factor (sometimes 20 = .95 is better), with LrComp = .1 comp compensating for increased effective learning rate.
    + `DWt = Lrate * dwt`
        + final effective weight change includes overall learning rate multiplier.  For learning rate schedules, just directly manipulate the learning rate parameter -- not using any kind of builtin schedule mechanism.

* **Weight Balance** -- this option (off by default but recommended for larger models) attempts to maintain more balanced weights across units, to prevent some units from hogging the representational space, by changing the rates of weight increase and decrease in the soft weight bounding function, as a function of the average receiving weights.  All params in `WtBalParams`:
    + `if (Wb.Avg < LoThr): Wb.Fact = LoGain * (LoThr - MAX(Wb.Avg, AvgThr)); Wb.Dec = 1 / (1 + Wb.Fact); Wb.Inc = 2 - Wb.Dec`
    + `else: Wb.Fact = HiGain * (Wb.Avg - HiThr); Wb.Inc = 1 / (1 + Wb.Fact); Wb.Dec = 2 - Wb.Inc`
        + `Wb` is the `WtBalRecvPrjn` structure stored on the `leabra.Prjn`, per each Recv neuron.  `Wb.Avg` = average of recv weights (computed separately and only every N = 10 weight updates, to minimize computational cost).  If this average is relatively low (compared to LoThr = .4) then there is a bias to increase more than decrease, in proportion to how much below this threshold they are (LoGain = 6).  If the average is relatively high (compared to HiThr = .4), then decreases are stronger than increases, HiGain = 4.
    + A key feature of this mechanism is that it does not change the sign of any weight changes, including not causing weights to change that are otherwise not changing due to the learning rule.  This is not true of an alternative mechanism that has been used in various models, which normalizes the total weight value by subtracting the average.  Overall this weight balance mechanism is important for larger networks on harder tasks, where the hogging problem can be a significant problem.

* **Weight update equation** 
    + The `LWt` value is the linear, non-contrast enhanced version of the weight value, and `Wt` is the sigmoidal contrast-enhanced version, which is used for sending netinput to other neurons.  One can compute LWt from Wt and vice-versa, but numerical errors can accumulate in going back-and forth more than necessary, and it is generally faster to just store these two weight values.
    + `DWt *= (DWt > 0) ? Wb.Inc * (1-LWt) : Wb.Dec * LWt`
        + soft weight bounding -- weight increases exponentially decelerate toward upper bound of 1, and decreases toward lower bound of 0, based on linear, non-contrast enhanced LWt weights.  The `Wb` factors are how the weight balance term shift the overall magnitude of weight increases and decreases.
    + `LWt += DWt`
        + increment the linear weights with the bounded DWt term
    + `Wt = SIG(LWt)`
        + new weight value is sigmoidal contrast enhanced version of linear weight 
        + `SIG(w) = 1 / (1 + (Off * (1-w)/w)^Gain)`
    + `DWt = 0`
        + reset weight changes now that they have been applied.


