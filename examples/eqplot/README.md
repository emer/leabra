# EqPlot

This simple sim models a double exponential synapse, and plots the results in a etable.Table and eplot.Plot2D. This is a good example to copy for plotting equations to explore how they respond to parameter changes, etc.

The equation for the biexponential synapse (with separate rise and decay time constants) was copied from the [Brian2](https://brian2.readthedocs.io/en/stable/user/converting_from_integrated_form.html) documentation, which kindly provides the translation from the integral form (as a product of exponentials) into ODE form which we can use to incrementally update over time.



