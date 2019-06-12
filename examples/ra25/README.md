# Leabra Random Associator 25 Example

This example project serves as a demo and basic template for Leabra models -- you can use this as a starting point for creating your own models.  It has been constructed to illustrate the most common types of functionality needed across models, to facilitate copy-paste programming.

# Running the model

First, see [Wiki Install](https://github.com/emer/emergent/wiki/Install) for installation instructions, which includes how to build the model code in this directory, which will make an executable named `ra25` that you run from a terminal command line:

```bash
./ra25
```

You can also run the [Python](https://github.com/emer/leabra/blob/master/python/README.md) version of the model by following those instructions.

## Basic running and graphing

The basic interface has a toolbar across the top, a set of `Sim` fields on the left, and a tab bar with the network view and various log graphs on the right.  The toolbar actions all have tooltips so just hover over those to see what they do.  To speed up processing, you'll want to either turn off the `ViewOn` flag or switch to a graph tab instead of the `NetVew` tab.

## Testing

The Test* buttons allow you to test items, including a specifically-chosen item, without having any effect on the ongoing training.  This is one advantage of the [Env](https://github.com/emer/emergent/wiki/Env) interface, which keeps all the counters associated with training and testing separate.

The NetView will show cycle-by-cycle updates during testing, and you can see the temporal evolution of the activities in the `TstCycPlot`.  If you do `TestAll` and look at the `TstTrlPlot` you can see the current performance on every item.  Meanwhile, if you click on the `TstTrlLog` button at the left, you can see the input / output activations for each item in a TableView, and the `TstErrLog` button likewise shows the same thing but filtered to only show those trials that have an error.  `TstErrStats` computes some stats on those error trials -- not super meaningful here but could be in other more structured environments, and the code that does all this shows how to do all of this kind of data analysis using the [etable.Table](https://github.com/emer/etable) system, which is similar to the widely-used pandas DataFrame structure in Python, and is the updated version of the `DataTable` from C++ emergent.

## Parameter searching

Clicking on the `Params` button will pull up a set of parameters, the design and use of which are explained in detail on the wiki page: [Params](https://github.com/emer/emergent/wiki/Params).  When you hit `Init`, the `Base` ParamSet is always applied, and then if you enter the name of another ParamSet in the `ParamSet` field, that will then be applied after the Base, thereby overwriting those base default params with other ones to explore.

To see any non-default parameter settings, the `Non Def Params` button in the NetView toolbar will show you those, and the `All Params` button will show you *all* of the parameters for every layer and projection in the network.  This is a good way to see all the parameters that are available.

To determine the real performance impact of any of the params, you typically need to collect stats over multiple runs.  To see how this works, try the following:

* Click on the `RunPlot` tab and hit `ResetRunLog` for good measure.
* Init with `ParamSet` = empty, and do `Train` and let it run all 10 runs.  By default, it plots the epoch at which the network first hit 0 errors (`FirstZero`), which is as good a measure as any of overall learning speed.
* When it finishes, you can click on the `RunStats` Table to see the summary stats for FirstZero and the average over the last 10 epochs of `PctCor`, and it labels this as using the Base params.
* Now enter `NoMomentum` in the `ParamSet`, `Init` and `Train` again.  Then click on the `RunStats` table button again (it generates a new one after every complete set of runs, so you can just close the old one -- it won't Update to show new results).  You can now directly compare e.g., the Mean FirstZero Epoch, and see that the `Base` params are slightly faster than `NoMomentum`.
* Now you can go back to the params, duplicate one of the sets, and start entering your own custom set of params to explore, and see if you can beat the Base settings!  Just click on the `*params.Sel` button after `Network` to get the actual parameters being set, which are contained in that named `Sheet`.
* Click on the `Net` button on the left and then on one of the layers, and so-on into the parameters at the layer level (`Act`, `Inhib`, `Learn`), and if you click on one of the `Prjn`s, you can see parameters at the projection level in `Learn`.  You should be able to see the path for specifying any of these params in the Params sets.
* We are planning to add a function that will show you the path to any parameter via a context-menu action on its label..

## Running from command line

Type this at the command line:
```bash
./ra25 -help
```

To see a list of args that you can pass -- passing any arg will cause the model to run without the gui, and save log files and, optionally, final weights files for each run.

# Code organization and notes

Most of the code is commented and should be read directly for how to do things.  Here are just a few general organizational notes about code structure overall.

* Good idea to keep all the code in one file so it is easy to share with others, although fine to split up too if it gets too big -- e.g., logging takes a lot of space and could be put in a separate file.

* In Go, you can organize things however you want -- there are no constraints on order in Go code.  In Python, all the methods must be inside the main Sim class definition but otherwise order should not matter.

* The GUI config and elements are all optional and the -nogui startup arg, along with other args, allows the model to be run without the gui.

* If there is a more complex environment associated with the model, always put it in a separate file, so it can more easily be re-used across other models.

* The params editor can easily save to a file, default named "params.go" with name `SavedParamsSets` -- you can switch your project to using that as its default set of params to then easily always be using whatever params were saved last.



