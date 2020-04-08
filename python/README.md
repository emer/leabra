# Python interface to Emergent

You can now run the Go version of *emergent* via Python, using a newly-updated version of the [gopy](https://github.com/go-python/gopy) tool that automatically creates Python bindings for Go packages. 

See the [GoGi Python README](https://github.com/goki/gi/blob/master/python/README.md) for more details on how the python wrapper works and how to use it for GUI-level functionality.

Note: **You must follow the installation instructions in the above GoGi Python README** to install the `gopy` program prior to running the further installation instructions below.  Given that emergent depends fully on GoGi, doing this first ensures that everything is all working prior to moving on to emergent itself.

There is a Python version of the basic leabra demo in `examples/ra25/ra25.py`, which you can compare with the `ra25.go` to see how similar the python and Go code are.  While it is possible to use standard Python data structures such as `pandas` for the input / output patterns presented to the network, and recording a log of the results, there is extra GUI support for the Go-based `etable.Table`, so we are using that in the Python version as well.  We will include optimized functions for efficiently converting between the etable.Table and pandas and other such table-like data structures, but for now, you can simply save your data to a .csv and load it from there, to do further data analysis etc using your favorite workflow, etc.

Because the Go and Python versions of this ra25 demo are essentially identical in function, the README (click on the button in the toolbar when the program runs) instructions apply to both.

# Installation

First, you have to install the Go version of emergent: [Wiki Install](https://github.com/emer/emergent/wiki/Install), and follow the [GoGi Python README](https://github.com/goki/gi/blob/master/python/README.md) installation instructions, and make sure everything works with the standard GoGi `widgets` example.

Python version 3 (3.6 has been well tested) is recommended.

```sh
$ cd ~/go/src/github.com/emer/leabra/python    # or $GOPATH if go not in ~/go
$ make
$ make install  # may need to do sudo make install -- installs into /usr/local/bin and python site-packages
$ cd ../examples/ra25
$ pyleabra -i ra25.py   # pyleabra was installed during make install into /usr/local/bin
```

* The `pyleabra` executable combines standard python and the full Go emergent and GoGi gui packages -- see the information in the GoGi python readme for more technical information about this.


