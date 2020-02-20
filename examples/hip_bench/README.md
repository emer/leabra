This project supports testing of the hippocampus model, systematically varying different parameters (number of patterns, sizes of different layers, etc), and recording results.

It is both for optimizing parameters and also testing new learning ideas in the hippocampus.

# Best Params for AB-AC, Feb 2020

Dramatic improvements in learning performance were achieved by optimizing the following parameters and adding the following mechanisms:

## Error-driven CA3

Modified `AlphaCyc` to reduce the strength of DG -> CA3 mossy inputs for the first quarter, then increase back to regular strength in Q2 onward.  This creates a minus phase state in CA3 in ActQ1, where it is driven primarily / exclusively by its ECin -> CA3 inputs.  By contrasting with final ActP state, this drives std error-driven learning in all the CA3 projections.

The best params were a WtScale.Rel = 4 for mossy inputs, which is reduced to 0 during Q1, by setting MossyDel=4.  This is in contrast to the Rel = 8 used in default params.

While theoretically interesting, this is not the most critical performance factor overall.

## Reduced DG on Test

Decreasing the DG -> CA3 input during test significantly improves performance overall -- setting MossyDelTest = 3 was best (going all the way to 4 was significantly worse).  This allows the EC -> CA3 pathway to dominate more during testing, supporting more of a pattern-completion dynamic.  This is also closer to what the network experiences during error-driven learning.

## Strong ECin -> DG learning

ECin -> DG is playing a critical role in learning overall, and benefits from a high, fast learning rate.  In effect, it is stamping-in a specific pattern for each DG unit, and potentially separating the units further through this strong Hebbian learning which, using the CPCA mode, is turning off inactive inputs.  This ability to turn off inactive inputs also seems to be important for CA3 -> CA1, which works better with CPCA than BCM hebbian.

However, learning in the DG -> CA3 pathway (mossies) is definitely bad

## Reduced CA3 <-> CA3 recurrents

Reducing strength of recurrent collaterals in CA3 improves performance significantly.  Also reducing learning rate to .1 instead of .15 for main perforant path inputs.

## Somewhat sparser mossy inputs

Reducing MossyPCon = .02 instead of .05 was better, but not further.

## Adding BCM Hebbian to EC <-> CA1

The standard Leabra BCM hebbian learning works better than the hip.CHLPrjn CPCA Hebbian learning.

