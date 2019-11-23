# PBWM

[![GoDoc](https://godoc.org/github.com/emer/leabra/pbwm?status.svg)](https://godoc.org/github.com/emer/leabra/pbwm)

PBWM is the *prefrontal-cortex basal-ganglia working memory* model [O'Reilly & Frank, 2006](#references), where the basal ganglia (BG) drives *gating* of PFC working memory maintenance, switching it dynamically between *updating* of new information vs. *maintenance* of existing information.  It was originally inspired by existing data, biology, and theory about the role of the BG in motor action selection, and the LSTM (long short-term-memory) computational model of [Hochreiter & Schmidhuber](#references), which solved limitations in existing recurrent backpropagation networks by adding dynamic input and output gates.  These LSTM models have experienced a significant resurgence along with the backpropagation neural networks in general.

The simple computational idea is that the BG gating signals fire phasically to disinhibit corticothalamic loops through the PFC, enabling the robust maintenance of new information there.  In the absence of such gating signals, the PFC will continue to maintain existing information.

In the current version, we integrate with the broader `DeepLeabra` framework (in the `deep` directory) that incorporates the separation between *superficial* and *deep* layers in cortex and their connections with the thalamus: the thalamocortical loops are principally between the deep layers.  Thus, within a given PFC area, you can have the superficial layers being more sensitive to current inputs, while the deep layers are more robustly maintaining information through the thalamocortical loops.

Furthermore, it allows a unification of maintenance and output gating, both of which are effectively opening up a gate between superficial to deep (via the thalamocortical loops) -- deep layers the drive principal output of frontal areas (e.g., in M1, deep layers directly drive motor actions through subcortical projections).  In PFC, deep layers are a major source of top-down activation to other cortical areas, in keeping with the idea of executing "cognitive actions" that influence processing elsewhere in the brain.  The only real difference is whether the neurons exhibit significant sustained maintenance, or are only phasically activated by gating.  Both profiles are widely observed e.g., [Sommer & Wurtz, 2000](#references).

The key, more complex computational challenges are:

* How to actually sequence the updating of PFC from maintaining prior information to now encoding new information, which requires some mechanism for *clearing* out the prior information.

* How *maintenance* and *output* gating within a given area are organized and related to each other.

* Learning *when* the BG should drive update vs. maintain signals, which is particularly challenging because of the temporally-delayed nature of the consequence of an earlier gating action -- you only know if it was useful to maintain something *later* when you need it.  This is the *temporal credit assignment* problem.

## Updating

For the updating question, we compute a BG gating signal in the middle of quarter 1 of the overall AlphaCycle of processing (cycle 18 of 25), which has the immediate effect of clearing out the existing activations, such that by the end of the next quarter (2), the new information is sufficiently represented in the superficial PFC neurons.  At the end of quarter 2, the superficial activations drive updating of the deep layers (via the standard `deep` `CtxtGe` computation), to maintain the new information.  In keeping with the beta frequency cycle of the BG / PFC circuit (20 Hz, 50 msec cycle time), we support a second round of gating in the middle of quarter 2, followed by maintenance activating in deep layers after quarter 4.  In summary:

* Q1, cycle 18: BG gating, PFC clearing of any existing act
* Q2, end: Super -> Deep (CtxtGe)
* Q2, cycle 18: BG gating, PFC clearing
* Q4, end: Super -> Deep (CtxtGe)

## Maint & Output Organization

For the organization of `Maint` and `Out` gating, we make the simplifying assumption that each hypercolumn ("stripe") of maintenance PFC has a corresponding output stripe, so you can separately decide to maintain something for an arbitrary amount of time, and subsequently use that information via output gating.  A key question then becomes: what happens to the maintained information?  Empirically, many studies show a sudden termination of active maintenance at the point of an action using maintained information [Sommer & Wurtz, 2000](#references), which makes computational sense: "use it and lose it".  In addition, it is difficult to come up with a good positive signal to independently drive clearing: it is much easier to know when you do need information, than to know the point at which you no longer need it.  Thus, we have output gating clear corresponding maintenance gating (there is an option to turn this off too, if you want to experiment).  The availability of "open" stripes for subsequent maintenance after this clearing seems to be computationally beneficial in our tests.

## Learning

Finally, for the learning question, we adopt a computationally powerful form of *trace-based* dopamine-modulated learning, where each BG gating action leaves a synaptic trace, which is finally converted into a weight change as a function of the next phasic dopamine signal, providing a summary "outcome" evaluation of the net value of the recent gating actions.  This directly solves the temporal credit assignment problem, by allowing the synapses to bridge the temporal gap between action and outcome, over a reasonable time window, with multiple such gating actions separately encodable.

Biologically, we suggest that widely-studied synaptic tagging mechanisms have the appropriate properties for this trace mechanism.  Extensive research has shown that these synaptic tags, based on actin fiber networks in the synapse, can persist for up to 90 minutes, and when a subsequent strong learning event occurs, the tagged synapses are also strongly potentiated ([Redondo & Morris, 2011, Rudy, 2015, Bosch & Hayashi, 2012](#references)). 

This form of trace-based learning is very effective computationally, because it does not require any other mechanisms to enable learning about the reward implications of earlier gating events.  In earlier versions of the PBWM model, we relied on CS (conditioned stimulus) based phasic dopamine to reinforce gating, but this scheme requires that the PFC maintained activations function as a kind of internal CS signal, and that the amygdala learn to decode these PFC activation states to determine if a useful item had been gated into memory. 

# Layers

Here are the details about each different layer type in PBWM:

*  **Matrix**: this is the dynamic gating system representing the matrix units within the dorsal striatum of the basal ganglia.  The `MatrixGo` layer contains the "Go" (direct pathway) units, while the `MatrixNoGo` layer contains "NoGo" (indirect pathway).  The Go units, expressing more D1 receptors, increase their weights from dopamine bursts, and decrease weights from dopamine dips, and vice-versa for the NoGo units with more D2 receptors. As is more consistent with the BG biology than earlier versions of this model, most of the competition to select the final gating action happens in the GPe and GPi (with the hyperdirect pathway to the subthalamic nucleus also playing a critical role, but not included in this more abstracted model), with only a relatively weak level of competition within the Matrix layers.   We also combine the maintenance and output gating stripes all in the same Matrix layer, which allows them to all compete with each other here, and more importantly in the subsequent GPi and GPe stripes.  This competitive interaction is critical for allowing the system to learn to properly coordinate maintenance when it is appropriate to update/store new information for maintenance vs. when it is important to select from currently stored representations via output gating.
	
*  **GPeNoGo:** provides a first round of competition between all the NoGo stripes, which critically prevents the model from driving NoGo to *all* of the stripes at once. Indeed, there is physiological and anatomical evidence for NoGo unit collateral inhibition onto other NoGo units. Without this NoGo-level competition, models frequently ended up in a state where all stripes were inhibited by NoGo, and when *nothing* happens, *nothing* can be learned, so the model essentially fails at that point!

*  **GpiThal:** Has a strong competition for selecting which stripe gets to gate, based on projections from the MatrixGo units, and the NoGo influence from GPeNoGo, which can effectively *veto* a few of the possible stripes to prevent gating. As discussed in the BG model, here we have combined the functions of the GPi (or SNr) and the Thalamus into a single abstracted layer, which has the excitatory kinds of outputs that we would expect from the thalamus, but also implements the stripe-level competition mediated by the GPi/SNr. If there is more overall Go than NoGo activity, then the GPiThal unit gets activated, which then effectively establishes an excitatory loop through the corresponding deep layers of the PFC, with which the thalamus neurons are bidirectionally interconnected. 

* **Rew, RWPred, SNc:** The `Rew` layer represents the reward activation driven on the Recall trials based on whether the model gets the problem correct or not, with either a 0 (error, no reward) or 1 (correct, reward) activation.  `RWPred` is the prediction layer that learns based on dopamine signals to predict how much reward will be obtained on this trial.  The **SNc** is the final dopamine unit activation, reflecting reward prediction errors. When outcomes are better (worse) than expected or states are predictive of reward (no reward), this unit will increase (decrease) activity. For convenience, tonic (baseline) states are represented here with zero values, so that phasic deviations above and below this value are observable as positive or negative activations. (In the real system negative activations are not possible, but negative prediction errors are observed as a pause in dopamine unit activity, such that firing rate drops from baseline tonic levels). Biologically the SNc actually projects dopamine to the dorsal striatum, while the VTA projects to the ventral striatum, but there is no functional difference in this level of model.


# Detailed Timing

## Quarter and longer

`GateState.Cnt` provides key tracker of gating state:
* -1 = initialized to this value, not maintaining
* 0 = just gated – any time the thal activity exceeds the gating threshold we reset counter (re-gate)
* >= 1: maintaining – first gating goes to 1 in Quarter_Init just following the gating quarter, counts up thereafter.
* <= -1: not maintaining – when cleared, reset to -1 in Quarter_Init just following clearing quarter, counts down thereafter.


| Trial.Qtr | Phase | BG                       | PFCmnt                                          | PFCmntD                                                    | PFCout                                                 | PFCoutD                                                    | Notes                       a                    |
|-----------|-------|--------------------------|-------------------------------------------------|--------------------------------------------------------------|--------------------------------------------------------|--------------------------------------------------------------|-------------------------------------------------|
| 1.1       | init  |                          |                                                 |                                                              |                                                        |                                                              |                                                 |
| 1.1       | --    | GPi Go -&gt; PFCmnt.thal | input -&gt; act                                 | Ctxt = 0, no act                                       | input -&gt; act                                        | Ctxt = 0, no act                                       | cortico-cortical super only, gate mid-quarter   |
| 1.2       | --    |                          | Burst = (act &gt; thr) * thal_eff         | "                                                            | "                                                      | "                                                            | mnt super d5b gets gating signal                |
| 1.3       | init  |                          | thal &gt; 0: ThalCnt++                        | s Burst -&gt; Ctxt, Ctxt &gt; 0: ThalCnt++ |                                                        | "                                                            | mnt deep gets d5b gated "context"               |
| 1.3       | --    |                          | DeepLrn + input -&gt; act                     | Ctxt -&gt; act -&gt; s DeepLrn; out?; trc            | gets mnt deep input (opt)                              | "                                                            | mnt continues w/ no out gating, out gets primed |
| 1.4       | +     | Other Go                 | Burst = (act &gt; thr) * thal_eff         | trc d5b clamp -&gt; netin, learning                          | "                                                      | "                                                            | TRC auto-encoder learning for mnt, via deep     |
| 2.1       | init  |                          | ThalCnt &gt; 0: ThalCnt++                   | s Burst -&gt; Ctxt, Ctxt &gt; 0: ThalCnt++ |                                                        | "                                                            | mnt continues                                   |
| 2.1       | --    |                          | "                                               | "                                                            | "                                                      | "                                                            |                                                 |
| 2.2       | --    | GPi Go -&gt; PFCout.thal | "                                               | "                                                            | Burst = (act &gt; thr) * thal_eff                | "                                                            | out super d5b gets gating signal                |
| 2.3       | init  |                          | ThalCnt &lt; 0: cleared fm out, Burst = 0 | Ctxt = 0, ThalCnt = -1                               | thal &gt; 0: ThalCnt++                               | s Burst -&gt; Ctxt, Ctxt &gt; 0: ThalCnt++ | out deep gets d5b gating                        |
| 2.3       | --    |                          | input -&gt; act                                 | Ctxt = 0, no act                                       | DeepLrn + input -&gt; act                            | Ctxt -&gt; act -&gt; s DeepLrn; output; trc          | out gating takes effect, driving actual output  |
| 2.4       | +     | Other Go                 | "                                               | "                                                            | "                                                      | "                                                            | continued output driving                        |
| 3.0       | init  |                          | ThalCnt &lt; 0: ThalCnt--                   | ThalCnt &lt; 0: ThalCnt--                                | ThalCnt &gt; out_mnt: ThalCnt = -1, Burst = 0 | Ctxt = 0: ThalCnt = -1                               | out gating cleared automatically after 1 trial  |


## Cycle

### C++ version

* Gating Cyc:
    + Cycle() -- usual
    + GateSend(): GPiThalLayer: detects gating, sends GateState signals to all who respond *next cycle*

* Cyc+1: 
    + ComputeAct:
        + Matrix: PatchShunt, SaveGatingThal, OutAChInhib in ApplyInhib
        + PFC:    PFCGating

# Details

* Currently only supporting 1-to-1 Maint and Out prjns -- Out gating automatically clears same pool in maint
        
# TODO

- [ ] Matrix uses net_gain = 0.5 -- why??  important for SIR2?, but not SIR1

- [ ] patch -- not *essential* for SIR1, test in SIR2

- [ ] TAN -- not *essential* for SIR1, test for SIR2

- [ ] del_inhib -- delta inhibition -- SIR1 MUCH faster learning without!  test for SIR2

- [ ] slow_wts -- not important for SIR1, test for SIR2

- [ ] GPe, GPi learning too -- allows Matrix to act like a hidden layer!

# References

Bosch, M., & Hayashi, Y. (2012). Structural plasticity of dendritic spines. Current Opinion in Neurobiology, 22(3), 383–388. https://doi.org/10.1016/j.conb.2011.09.002

Hochreiter, S., & Schmidhuber, J. (1997). Long Short Term Memory. Neural Computation, 9, 1735–1780.

O'Reilly, R.C. & Frank, M.J. (2006), Making Working Memory Work: A Computational Model of Learning in the Frontal Cortex and Basal Ganglia. Neural Computation, 18, 283-328.

Redondo, R. L., & Morris, R. G. M. (2011). Making memories last: The synaptic tagging and capture hypothesis. Nature Reviews Neuroscience, 12(1), 17–30. https://doi.org/10.1038/nrn2963

Rudy, J. W. (2015). Variation in the persistence of memory: An interplay between actin dynamics and AMPA receptors. Brain Research, 1621, 29–37. https://doi.org/10.1016/j.brainres.2014.12.009

Sommer, M. A., & Wurtz, R. H. (2000). Composition and topographic organization of signals sent from the frontal eye field to the superior colliculus. Journal of Neurophysiology, 83(4), 1979–2001.


