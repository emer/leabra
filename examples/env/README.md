# Example Environment

This project provides a simple example of how to use the [Env](https://github.com/emer/emergent/wiki/Env) interface to generate inputs for a network.

The `deep_fsa` and `sir2` projects also have Env implementations, but they are more complex, so this example may provide a more generic starting point for building your own custom Env.

The Env code is entirely in `env.go`, and `sim.go` is just the `ra25` simulation code adapted to use the env.

The env just activates a point in a 2D input matrix and has one-hot output representations for the X and Y coordinates of that point.

See also https://github.com/CompCogNeuro/sims for other models with Env implementations, including `ch9/sg` which uses the general-purpose `emergent/esg` stochastic generator to generate sentence inputs.

