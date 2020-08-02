# ATTrn: Attentional TRN model

The overall framework here is to support wide-scale inhibition that is much like what we have currently in a 4D layer with both layer-level and pool-level inhibition, but with a much more natural and flexible broad-scale computation that spans across layers, based on the structure of the TRN (thalamic reticular nucleus).  Thus, the primary layers in the network should only have Pool-based "local" cortical inhibition in place, and their broader inhibition will be supplied through these attentional mechanisms.

The pool then becomes the key organizing principle for network-level competition, much like a mixtures-of-experts framework, where the TRN plays the role of the gating layer.  Frontal / BG control networks have special abilities to bias pool-level competition in the TRN, providing an extra level of attentional control.

This package is implemented separately here, to more clearly isolate the code, but is included and primarily used within the `deep` DeepLeabra package. To make this work with deep leabra, we need to reorganize the CT / Pulvinar projections, to preserve the closed-loop characteristic of CT -> TRC -> L4 pathways.  This means that the drivers need to be dispersed, which presents extra difficulties, but is necessary.  So we need a TRC "packager" that takes all the things that a given pool participates in predicting, and packs them into a sequence of units in a pool.  Just keep an offset and do it 1D.  Will also need to have current TRC matched to driver layer, for stats and visualization..

* `AttnLayer` has an Attn variable, which can optionally multiply the Ge, Act, and ActLrn variables.  The ActLrn effects are typically the strongest -- learning should be most sensitive to attentional modulation.  Biologically, Attn corresponds to the *differential* strength in output of TRC neurons, as a result of TRN pooled inhibition.  From perspective of a superficial layer neuron, excitatory input from L4 is decreased in proportion to this inhibition, so it can affect *feedforward* Ge.  Ge could just be enough but also may want direct effects on Act, and reductions in dendritic drive could explain ActLrn effects (in addition to effects on Pulvinar / TRC).

    + Feedforward selective effects require operating on raw integration of Ge input signals, to only apply to Forward projections.

* `TRNLayer` is the summary attentional state of the TRC / TRN, *at the pool level*, which then drives the Attn signal for the AttnLayer (by name).  It receives summary excitatory drive from corresponding CT and TRC layers, derived directly from Pool.Inhib.Acts.Avg (i.e., FB inhib driver).

   + Important to emphasize that TRNLayer is more accurately the TRC pool-level activation as driven by TRN inhibition -- TRN proper is just the (not explicitly simulated!) inhibitory neurons!

* CT inputs are like FF inhib, TRC are like FB. But it is not clear how exactly to implement that in TRN..  Keep in mind tho.

## How to structure the computation

* Within-area (layer) competition should be stronger than between.  This is supported by TRNLayer being a single layer, mapping onto a corresponding layer at the pool-reduced level.  Keeping each TRNLayer separate and directly mapped is convenient.

* Broadest pooling of inhibition is like interinhib (layer group) -- share Max Gi..

* A given area (esp V1) may have multiple different resolutions of coverage of same area.  

```
. . . . . . . .
. . . . . . . .
```

# Compounding: Getting the Good without too much Lock-In

It is relatively easy to make something that locks in a given attentional pattern, but a problem arises when you then need to change things in response to new inputs -- often the network suffers from too much attentional lock-in...


# Reynolds & Heeger (2009)


# Folded Feedback (Grossberg, 1999)

Grossberg (1999) emphasized that it can be beneficial for attention to modulate the *inputs* to a given area, so it gets "folded" into the input stream.  Another way of thinking about this is that it is more effective to block a river further upstream, before further "compounding" effects might set in, rather than waiting until everything has piled in and you have to push against a torrent.   This is achieved by modulating the layer 4 inputs to an area, which happens by modulating forward projections.



