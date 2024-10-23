# PBWM

See [sir2](examples/sir2) example for working model.

In the [cemer](https://github.com/emer/cemer) (C++ emergent) versioning system, this is **version 5** of PBWM, reflecting a number of intermediate unpublished versions.

As of 9/2020, this implementation has been separated from the `deep` code, and there is just a `PFCDeepLayer` that manages all of the PFC-specific functionality, grabbing values from corresponding neurons in the Super layer, which can be of any type.  Also, the `rl` dopamine component is now independent and any source of dopamine can be used.  Current development in this space of ideas is taking place in the [axon](https://github.com/emer/axon) framework, so this `pbwm` package represents its own separate stable branch going forward.

PBWM is the *prefrontal-cortex basal-ganglia working memory* model [O'Reilly & Frank, 2006](#references), where the basal ganglia (BG) drives *gating* of PFC working memory maintenance, switching it dynamically between *updating* of new information vs. *maintenance* of existing information.  It was originally inspired by existing data, biology, and theory about the role of the BG in motor action selection, and the LSTM (long short-term-memory) computational model of [Hochreiter & Schmidhuber](#references), which solved limitations in existing recurrent backpropagation networks by adding dynamic input and output gates.  These LSTM models have experienced a significant resurgence along with the backpropagation neural networks in general.

The simple computational idea is that the BG gating signals fire phasically to disinhibit corticothalamic loops through the PFC, enabling the robust maintenance of new information there.  In the absence of such gating signals, the PFC will continue to maintain existing information.  The output of the BG through the GPi (globus pallidus internal segment) and homologous SNr (substantia nigra pars reticulata) represents a major bottleneck with a relatively small number of neurons, meaning that each BG gating output affects a relatively large group of PFC neurons.  One idea is that these BG gating signals target different PFC hypercolumns or *stripes* -- these correspond to `Pools` of neurons within the layers in the current implementation.

There are two PFC lamina, `PFCSuper` and `PFCDeep`, which are inspired by the broader `DeepLeabra` framework (in the `deep` directory) that incorporates the separation between *superficial* and *deep* layers in cortex and their connections with the thalamus: the thalamocortical loops are principally between the deep layers.  Thus, within a given PFC area, you can have the superficial layers being more sensitive to current inputs, while the deep layers are more robustly maintaining information through the thalamocortical loops.  The deep layer neurons mirror those in the superficial layer (as in a microcolumn), and receive activation from the superficial layer only when gating occurs -- after that point their dynamics can be controlled by specific profiles (stable, rising, falling etc), with multiple possible such trajectories per superficial neuron.

The super vs. deep distinction allows a unification of maintenance and output gating, both of which are effectively opening up a gate between superficial to deep (via the thalamocortical loops) -- deep layers the drive principal output of frontal areas (e.g., in M1, deep layers directly drive motor actions through subcortical projections).  In PFC, deep layers are a major source of top-down activation to other cortical areas, in keeping with the idea of executing "cognitive actions" that influence processing elsewhere in the brain.  The only real difference is whether the neurons exhibit significant sustained maintenance, or are only phasically activated by gating.  Both profiles are widely observed e.g., [Sommer & Wurtz, 2000](#references).

The key, more complex computational challenges are:

* How to actually sequence the updating of PFC from maintaining prior information to now encoding new information, which requires some mechanism for *clearing* out the prior information.

* How *maintenance* and *output* gating within a given area are organized and related to each other.

* Learning *when* the BG should drive update vs. maintain signals, which is particularly challenging because of the temporally delayed nature of the consequence of an earlier gating action -- you only know if it was useful to maintain something *later* when you need it.  This is the *temporal credit assignment* problem.

## Updating

For the updating question, we compute a BG gating signal in the middle of the 1st quarter (set by `GPiThal.Timing.GateQtr`) of the overall AlphaCycle of processing (cycle 18 of 25, per `GPiThal.Timing.Cycle` parameter), which has the immediate effect of clearing out the existing PFC activations (see `PFCLayer.Maint.Clear`, such that by the end of the next quarter (2), the new information is sufficiently represented in the superficial PFC neurons.  At the end of 2nd quarter (per `PFCLayer.Gate.GateQtr`, which is typically 1 quarter after the GPiThal version), the superficial activations drive updating of the deep layers, to maintain the new information.  In keeping with the beta frequency cycle of the BG / PFC circuit (20 Hz, 50 msec cycle time), we support a second round of gating in the middle of the 2nd quarter (again by `GPiThal.Timing.GateQtr`), followed by maintenance activating in deep layers after the 4th quarter.

For `PFCout` layers (with `PFCLayer.Gate.OutGate` set), there is an `OutQ1Only` option (default true) which, with `PFCLayer.Gate.GateQtr` set to Q1, causes output gating to update at the end of the 1st quarter, which gives more time for it to drive output responding.   And the 2nd beta-frequency gating comes too late in a standard AlphaCycle based update sequence to drive output, so it is not useful.  However, supporting two phases of maintenance updating allows for stripes cleared by output gating (see next subsection) to update in the 2nd half of the alpha cycle, which is useful.

In summary, for `PFCmnt` maintenance gating:

* Q1, cycle 18: BG gating, PFC clearing of any existing act
* Q2, end: Super -> Deep
* Q2, cycle 18: BG gating, PFC clearing
* Q4, end: Super -> Deep

And `PFCout` output gating:

* Q1, cycle 18: BG gating -- triggers clearing of corresponding Maint stripe
* Q1, end: Super -> Deep, so Deep can drive network output layers

## Maint & Output Organization

For the organization of `Maint` and `Out` gating, we make the simplifying assumption that each hypercolumn ("stripe") of maintenance PFC has a corresponding output stripe, so you can separately decide to maintain something for an arbitrary amount of time, and subsequently use that information via output gating.  A key question then becomes: what happens to the maintained information?  Empirically, many studies show a sudden termination of active maintenance at the point of an action using maintained information [Sommer & Wurtz, 2000](#references), which makes computational sense: "use it and lose it".  In addition, it is difficult to come up with a good positive signal to independently drive clearing: it is much easier to know when you do need information, than to know the point at which you no longer need it.  Thus, we have output gating clear corresponding maintenance gating (there is an option to turn this off too, if you want to experiment).  The availability of "open" stripes for subsequent maintenance after this clearing seems to be computationally beneficial in our tests.

## Learning

Finally, for the learning question, we adopt a computationally powerful form of *trace-based* dopamine-modulated learning (in `MatrixTracePrjn`), where each BG gating action leaves a synaptic trace, which is finally converted into a weight change as a function of the next phasic dopamine signal, providing a summary "outcome" evaluation of the net value of the recent gating actions.  This directly solves the temporal credit assignment problem, by allowing the synapses to bridge the temporal gap between action and outcome, over a reasonable time window, with multiple such gating actions separately encodable.

Biologically, we suggest that widely studied synaptic tagging mechanisms have the appropriate properties for this trace mechanism.  Extensive research has shown that these synaptic tags, based on actin fiber networks in the synapse, can persist for up to 90 minutes, and when a subsequent strong learning event occurs, the tagged synapses are also strongly potentiated ([Redondo & Morris, 2011, Rudy, 2015, Bosch & Hayashi, 2012](#references)). 

This form of trace-based learning is very effective computationally, because it does not require any other mechanisms to enable learning about the reward implications of earlier gating events.  In earlier versions of the PBWM model, we relied on CS (conditioned stimulus) based phasic dopamine to reinforce gating, but this scheme requires that the PFC maintained activations function as a kind of internal CS signal, and that the amygdala learn to decode these PFC activation states to determine if a useful item had been gated into memory.  

The CS-driven DA under the trace-based framework effectively serves to reinforce sub-goal actions that lead to the activation of a CS, which in turn is predicting final reward outcomes.  Thus, the CS DA provides an intermediate bridging kind of reinforcement evaluating the set of actions leading up to that point.  Kind of a "check point" of success prior to getting the real thing.

# Layers

Here are the details about each different layer type in PBWM:

* [MatrixLayer](https://godoc.org/github.com/emer/leabra/pbwm#MatrixLayer): this is the dynamic gating system representing the matrix units within the dorsal striatum of the basal ganglia.  The `MatrixGo` layer contains the "Go" (direct pathway) units (`DaR = D1`), while the `MatrixNoGo` layer contains "NoGo" (indirect pathway, `DaR = D2`).  The Go units, expressing more D1 receptors, increase their weights from dopamine bursts, and decrease weights from dopamine dips, and vice-versa for the NoGo units with more D2 receptors. As is more consistent with the BG biology than earlier versions of this model, most of the competition to select the final gating action happens in the GPe and GPi (with the hyperdirect pathway to the subthalamic nucleus also playing a critical role, but not included in this more abstracted model), with only a relatively weak level of competition within the Matrix layers.   We also combine the maintenance and output gating stripes all in the same Matrix layer, which allows them to all compete with each other here, and more importantly in the subsequent GPi and GPe stripes.  This competitive interaction is critical for allowing the system to learn to properly coordinate maintenance when it is appropriate to update/store new information for maintenance vs. when it is important to select from currently stored representations via output gating.
	
* `GPeNoGo`:  This is a standard provides a first round of competition between all the NoGo stripes, which critically prevents the model from driving NoGo to *all* of the stripes at once. Indeed, there is physiological and anatomical evidence for NoGo unit collateral inhibition onto other NoGo units. Without this NoGo-level competition, models frequently ended up in a state where all stripes were inhibited by NoGo, and when *nothing* happens, *nothing* can be learned, so the model essentially fails at that point!

* [GPiThalLayer](https://godoc.org/github.com/emer/leabra/pbwm#GPiThalLayer): Has a strong competition for selecting which stripe gets to gate, based on projections from the MatrixGo units, and the NoGo influence from GPeNoGo, which can effectively *veto* a few of the possible stripes to prevent gating. We have combined the functions of the GPi (or SNr) and the Thalamus into a single abstracted layer, which has the excitatory kinds of outputs that we would expect from the thalamus, but also implements the stripe-level competition mediated by the GPi/SNr. If there is more overall Go than NoGo activity, then the GPiThal unit gets activated, which then effectively establishes an excitatory loop through the corresponding deep layers of the PFC, with which the thalamus neurons are bidirectionally interconnected.  This layer uses [GateLayer](https://godoc.org/github.com/emer/leabra/pbwm#GateLayer) framework to update [GateState](https://godoc.org/github.com/emer/leabra/pbwm#GateState) which is broadcast to the Matrix and PFC, so they have current gating state information.

* [PFCDeepLayer](https://godoc.org/github.com/emer/leabra/pbwm#PFCDeepLayer): Uses `deep`-like super vs. deep dynamics with gating (in `GateState` values broadcast from GPiThal) determining when deep updates from super.  Actual maintenance in deep layer can be set using `PFCDyn` fixed dynamics that provides a simple way of shaping a temporally evolving activation pattern over the layer, with a minimal case of just stable fixed maintenance.  The different dyn patterns take place in extra Y-axis rows, with the inner-loop being the super y axis, and outer loop being the different dyn types.  Whereas there was some ambiguity previously in whether the super layer reflected the maintenance in deep to some extent, or just the current input state, the current version makes a stronger distinction here: super is *only* representing the current sensory inputs.  This also avoids any need for explicit clearing of super prior to gating in new information.  Furthermore, the super layer can be any type of layer -- all of the specific functionality is now in the PFCDeepLayer type.

# Dopamine Layers

See the `rl` package for the `DALayer` interface and simpler forms of phasic dopamine algorithms, including Rescorla-Wagner and Temporal Differences (TD).

Given that PBWM minimally requires a RW-level "primary value" dopamine signal, basic models can use this as follows:

* **Rew, RWPred, SNc:** The `Rew` layer represents the reward activation driven on the Recall trials based on whether the model gets the problem correct or not, with either a 0 (error, no reward) or 1 (correct, reward) activation.  `RWPred` is the prediction layer that learns based on dopamine signals to predict how much reward will be obtained on this trial.  The **SNc** is the final dopamine unit activation, reflecting reward prediction errors. When outcomes are better (worse) than expected or states are predictive of reward (no reward), this unit will increase (decrease) activity. For convenience, tonic (baseline) states are represented here with zero values, so that phasic deviations above and below this value are observable as positive or negative activations. (In the real system negative activations are not possible, but negative prediction errors are observed as a pause in dopamine unit activity, such that firing rate drops from baseline tonic levels). Biologically the SNc actually projects dopamine to the dorsal striatum, while the VTA projects to the ventral striatum, but there is no functional difference in this level of model.

# Implementation Details

## Network

The [pbwm.Network](https://godoc.org/github.com/emer/leabra/pbwm#Network) provides "wizard" methods for constructing and configuring standard PBWM and RL components.

It extends the core `Cycle` method called every cycle of updating as follows:

```Go
func (nt *Network) Cycle(ltime *leabra.Time) {
	nt.Network.CycleImpl(ltime) // basic version from leabra.Network
	nt.GateSend(ltime)          // GateLayer (GPiThal) computes gating, sends to other layers
	nt.RecGateAct(ltime)        // Record activation state at time of gating (in ActG neuron var)

	nt.EmerNet.(leabra.LeabraNetwork).CyclePostImpl(ltime) // always call this after std cycle..
}
```

which determines the additional steps of computation after the activations have been updated in the current cycle, supporting the extra gating and DA modulation functions.

There is also a key addition to `QuarterFinal` method that calls `DeepMaint` where deep layers get their updated maintenance inputs from corresponding superficial layers (mediated through layer 5IB neurons in the biology, which burst periodically).  This is when the PFC layers update deep from super.

## GPiThal and GateState

[GPiThalLayer](https://godoc.org/github.com/emer/leabra/pbwm#GPiThalLayer) is source of key GateState:

`GateState.Cnt` provides key tracker of gating state.  It is separately updated in each layer -- GPiThal only broadcasts the basic `Act` signal and `Now` signals.  For PFC, `Cnt` is:
* `-1` = initialized to this value, not maintaining.
* `0` = just gated -- any time the GPiThal activity exceeds the gating threshold (at specified `Timing.Cycle`) we reset counter (re-gate)
* `>= 1`: maintaining -- first gating goes to 1 in `QuarterFinal` of the `BurstQtr` gating quarter, counts up thereafter.
* `<= -1`: not maintaining – when cleared, reset to -1 in Quarter_Init just following clearing quarter, counts down thereafter.

All gated PBWM layers are of type [GateLayer](https://godoc.org/github.com/emer/leabra/pbwm#GateLayer) which just has infrastructure to maintain `GateState` values and synchronize across layers.

## PFCLayer

[PFCDeepLayer](https://godoc.org/github.com/emer/leabra/pbwm#PFCDeepLayer) supports `mnt` and `out` types, and handles all the PFC-specific gating logic.

* Cycle: `ActFmG` calls `Gating`, which only does something when `GateState.Now = true`: it resets `Cnt = 0` indicating just gated.

* `QuarterFinal`: calls `DeepMaint` which grabs the corresponding superficial activation, if gating has happened, and otherwise it just updates the `MaintGe` conductance based on dynamics.

In summary, for `PFCmnt` maintenance gating:

* Q1, cycle 18: BG gating, PFC clearing of any existing act: `Gating` call
* Q2, end: Super -> Deep `DeepMaint` 
* ...

And `PFCout` output gating:

* Q1, cycle 18: BG gating -- triggers clearing of corresponding Maint stripe
* Q1, end: Super -> Deep so Deep can drive network output layers

# TODO

- [ ] Matrix uses net_gain = 0.5 -- why??  important for SIR2?, but not SIR1

- [ ] patch -- not *essential* for SIR1, test in SIR2

- [x] CIN -- did not work well at all surprisingly -- works much better in pcore..

- [ ] del_inhib -- delta inhibition -- SIR1 MUCH faster learning without!  test for SIR2

- [ ] slow_wts -- not important for SIR1, test for SIR2


# References

* Bosch, M., & Hayashi, Y. (2012). Structural plasticity of dendritic spines. Current Opinion in Neurobiology, 22(3), 383–388. https://doi.org/10.1016/j.conb.2011.09.002

* Hochreiter, S., & Schmidhuber, J. (1997). Long Short Term Memory. Neural Computation, 9, 1735–1780.

* O'Reilly, R.C. & Frank, M.J. (2006), Making Working Memory Work: A Computational Model of Learning in the Frontal Cortex and Basal Ganglia. Neural Computation, 18, 283-328.

* Redondo, R. L., & Morris, R. G. M. (2011). Making memories last: The synaptic tagging and capture hypothesis. Nature Reviews Neuroscience, 12(1), 17–30. https://doi.org/10.1038/nrn2963

* Rudy, J. W. (2015). Variation in the persistence of memory: An interplay between actin dynamics and AMPA receptors. Brain Research, 1621, 29–37. https://doi.org/10.1016/j.brainres.2014.12.009

* Sommer, M. A., & Wurtz, R. H. (2000). Composition and topographic organization of signals sent from the frontal eye field to the superior colliculus. Journal of Neurophysiology, 83(4), 1979–2001.


