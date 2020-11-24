# Python interface to emergent / Leabra

You can run the Go version of *emergent* via Python, using the [gopy](https://github.com/go-python/gopy) tool that automatically creates Python bindings for Go packages. 

See the [GoGi Python README](https://github.com/goki/gi/blob/master/python/README.md) for more details on how the python wrapper works and how to use it for GUI-level functionality.  **If you encounter any difficulties with this install, then try doing the install in GoGi first**, and read more about the different install issues there.

See the `.py` versions of various projects in `examples`, and especially in the [Comp Cog Neuro sims](https://github.com/CompCogNeuro/sims), for many examples of Python versions.  

See [etable pyet](https://github.com/emer/etable/tree/master/examples/pyet) for example code for converting between the Go `etable.Table` and `numpy`, `torch`, and `pandas` table structures.  All of the converted projects rely on `etable` because it provides a complete GUI interface for viewing and manipulating the data, but it is easy to convert any of these tables into Python-native formats (and copy back-and-forth).  The `pyet` python library (in `pyside` and auto-installed with this package) has the necessary routines.

# Installation

First, you have to install the Go version of emergent: [Wiki Install](https://github.com/emer/emergent/wiki/Install).

Python version 3 (3.6, 3.8 have been well tested) is recommended.

This assumes that you are using go modules, as discussed in the wiki install page, and *that you are in the `leabra` directory where you installed leabra* (e.g., `git clone https://github.com/emer/leabra` and then `cd leabra`)

```sh
$ cd python     # should be in leabra/python now -- i.e., the dir where this README.md is..
$ make
$ make install  # may need to do: sudo make install -- installs into /usr/local/bin and python site-packages
$ cd ../examples/ra25
$ ./ra25.py     # runs using magic code on first line of file -- alternatively:
$ pyleabra -i ra25.py   # pyleabra was installed during make install into /usr/local/bin
```

The `pyleabra` executable combines standard python and the full Go emergent and GoGi gui packages -- see the information in the GoGi python readme for more technical information about this.

