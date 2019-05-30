# Leabra Random Associator 25 Example

This example project serves as a demo and basic template for Leabra models.  It has been constructed to illustrate the most common types of functionality needed across models, to facilitate copy-paste programming.

# Running the model

## Basic running and graphing

## Testing

## Parameter searching

# Code organization and notes

Most of the code is commented and should be read directly for how to do things.  Here are just a few general organizational notes about code structure overall.

* Good idea to keep all the code in one file so it is easy to share with others, although fine to split up too if it gets too big -- e.g., logging takes a lot of space and could be put in a separate file.

* The GUI config and elements are all optional and the -nogui startup arg, along with other args, allows the model to be run without the gui. (**TODO**)

* If there is a more complex environment associated with the model, always put it in a separate file, so it can more easily be re-used across other models.

* The params editor can easily save to a file, default named "params.go" with name *??* -- you can switch your project to using that as its default set of params to then easily always be using whatever params were saved last.

# TODO

These are things that remain to be done:

- [x] err log -- demonstrates filtering

- [x] record layer activities in test trial and also do agg stats on them in errs

- [x] bug in structview: not updating -- saw that in full rebuild on marbles too -- think it might be due to removal of update code in style2d -- need to put that back in..

- [ ] hog stats

- [ ] select specific event to test

- [ ] net rel stats

- [ ] startup args and -nogui mode.

- [ ] neg draw for plots

- [ ] reset test epc data at start

- [ ] reset all data -- for tables that accumulate (run?)


