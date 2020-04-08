# knadapt

Package knadapt provides code for sodium (Na) gated potassium (K) currents that drive adaptation (accommodation) in neural firing.  As neurons spike, driving an influx of Na, this activates the K channels, which, like leak channels, pull the membrane potential back down toward rest (or even below).  Multiple different time constants have been identified and this implementation supports 3: M-type (fast), Slick (medium), and Slack (slow)

Here's a good reference:

Kaczmarek, L. K. (2013). Slack, Slick, and Sodium-Activated Potassium Channels.
ISRN Neuroscience, 2013. https://doi.org/10.1155/2013/354262

This package supports both spiking and rate-coded activations.

# Spiking

The logic is simplest for the spiking case:

```Go
	if spike {
		gKNa += Rise * (Max - gKNa)
	} else {
		gKNa -= 1/Tau * gKNa
	}
```

The KNa conductance ($g_{kna}$ in mathematical terminology, `gKNa` in the program) rises to a `Max` value with a `Rise` rate constant, when the neuron spikes, and otherwise it decays back down to zero with another time constant `Tau`.

# Rate code

The equivalent rate-code equation just substitutes the rate-coded activation variable in as a multiplier on the rise term:

```Go
	gKNa += act * Rise * (Max - gKNa) - (1/Tau * gKNa)
```

# Defaults
    
The default parameters, which were fit to various empirical firing patterns and also have proven useful in simulations, are:

| Channel Type     | Tau (ms) | Rise  |  Max  |
|------------------|----------|-------|-------|
| Fast (M-type)    | 50       | 0.05  | 0.1   |
| Medium (Slick)   | 200      | 0.02  | 0.1   |
| Slow (Slack)     | 1000     | 0.001 | 1.0   |


