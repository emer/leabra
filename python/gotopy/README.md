This directory has info and tools for converting Go-based simulation projects to Python.

# GoToPy

GoToPy does a first pass conversion of Go syntax to Python syntax: https://github.com/go-python/gotopy

```sh
$ go get github.com/go-python/gotopy
```

This does the install directly, so the gotopy executable should be in your `~/go/bin` directory, which you should add to your `PATH` if not already.  Check by typing: `which gotopy` for example.

To run directly:

```sh
$ gotopy -gogi mysim.go > mysim.py
```

The `-gogi` option is important for enabling extra conversions for the GoGi gui system.

# leabra-to.py

This Python program does additional steps specific to Leabra sims to attempt to get the code closer to usable.  To prepare, run:

```sh
$ pip3 install -r requirements.txt
```

You can copy this file to an appropriate place on your path, e.g.:

```sh
$ cp leabra-to.py /usr/local/bin
```

and then run it:

```sh
$ leabra-to.py mysim.go
```

which generates a file named `mysim.py` -- important: will overwrite any existing!

After running, you will need to fix the start and end by copying from an existing project that is similar (use ra25 if nothing else), with the CB callback functions at the top, and the `tbar.AddAction` calls in `ConfigGui` at the end that call these callbacks instead of the inline code.   There may be other errors which you can discover by running it -- there is a diminishing returns point on this conversion process so it is not designed to be complete.


