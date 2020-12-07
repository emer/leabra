# Benchmark results

5-layer networks, with same # of units per layer: SMALL = 25; MEDIUM = 100; LARGE = 625; HUGE = 1024; GINORM = 2048, doing full learning, with default params, including momentum, dwtnorm, and weight balance.

Results are total time for 1, 2, 4 threads, on my macbook.

## C++ Emergent

```
* Size     1 thr   2 thr    4 thr
---------------------------------
* SMALL:   2.383   2.248    2.042
* MEDIUM:  2.535   1.895    1.263
* LARGE:  19.627   8.559    8.105
* HUGE:   24.119  11.803   11.897
* GINOR:  35.334  24.768   17.794
```

## Go v1.15, 8/21/2020, leabra v1.1.5

Basically the same results as below, except a secs or so faster due to faster macbook pro. Layer.Act.Gbar.L = 0.2 instead of new default of 0.1 makes a *huge* difference!  

```
* Size     1 thr  2 thr  4 thr
---------------------------------
* SMALL:   1.27   3.53   3.64
* MEDIUM:  1.61   2.31   2.09
* LARGE:   9.56   7.48   5.44
* HUGE:   19.17   13.3   9.62
* GINOR:  23.61  17.94  13.24
```

```
$ ./bench -epochs 5 -pats 20 -units 625 -threads=1
Took  9.845 secs for 5 epochs, avg per epc:  1.969
TimerReport: BenchNet, NThreads: 1
    Function Name Total Secs    Pct
    ActFmG        1.824        18.59
    AvgMaxAct     0.09018       0.919
    AvgMaxGe      0.08463       0.8624
    CyclePost     0.002069      0.02108
    DWt           2.11         21.51
    GFmInc        0.3974        4.05
    InhibFmGeAct  0.107         1.091
    QuarterFinal  0.004373      0.04457
    SendGDelta    3.117        31.77
    WtBalFmWt     1.285e-05     0.0001309
    WtFmDWt       2.075        21.15
    Total         9.813
```

```
$ ./bench -epochs 5 -pats 10 -units 1024 -threads=1
Took  19.34 secs for 5 epochs, avg per epc:  3.868
TimerReport: BenchNet, NThreads: 1
    Function Name Total Secs    Pct
    ActFmG        1.639        8.483
    AvgMaxAct     0.07904      0.4091
    AvgMaxGe      0.07551      0.3909
    CyclePost     0.001287     0.006663
    DWt           3.669       18.99
    GFmInc        0.3667       1.898
    InhibFmGeAct  0.09876      0.5112
    QuarterFinal  0.004008     0.02075
    SendGDelta   10.21        52.87
    WtBalFmWt     1.2e-05      6.211e-05
    WtFmDWt       3.172       16.42
    Total        19.32
```

## Go emergent 6/2019 after a few bugfixes, etc: significantly faster!

```
* SMALL:   1.46   3.63   3.96
* MEDIUM:  1.87   2.46   2.32
* LARGE:  11.38   8.48   6.03
* HUGE:   19.53   14.52   11.29
* GINOR:  26.93   20.66   15.71
```

now really just as fast overall, if not faster, than C++!

note: only tiny changes after adding IsOff check for all neuron-level computation.

## Go emergent, per-layer threads, thread pool, optimized range synapse code

```
* SMALL:   1.486   4.297   4.644
* MEDIUM:  2.864   3.312   3.037
* LARGE:  25.09   20.06   16.88
* HUGE:   31.39   23.85   19.53
* GINOR:  42.18   31.29   26.06
```

also: not too much diff for wt bal off!

## Go emergent, per-layer threads, thread pool

```
* SMALL:  1.2180    4.25328  4.66991
* MEDIUM: 3.392145  3.631261  3.38302
* LARGE:  31.27893  20.91189 17.828935
* HUGE:   42.0238   22.64010  18.838019
* GINOR:  65.67025  35.54374  27.56567
```

## Go emergent, per-layer threads, no thread pool (de-novo threads)

```
* SMALL:  1.2180    3.548349  4.08908
* MEDIUM: 3.392145  3.46302   3.187831
* LARGE:  31.27893  22.20344  18.797924
* HUGE:   42.0238   29.00472  24.53498
* GINOR:  65.67025  45.09239  36.13568
```

# Per Function 

Focusing on the LARGE case:

C++: `emergent -nogui -ni -p leabra_bench.proj epochs=5 pats=20 units=625 n_threads=1`

```
BenchNet_5lay timing report:
function      time     percent 
Net_Input     8.91    43.1
Net_InInteg       0.71     3.43
Activation    1.95     9.43
Weight_Change 4.3     20.8
Weight_Updt       2.85    13.8
Net_InStats       0.177    0.855
Inhibition    0.00332  0.016
Act_post      1.63     7.87
Cycle_Stats       0.162    0.781
    total:       20.7
```

Go: `./bench -epochs 5 -pats 20 -units 625 -threads=1`

```
TimerReport: BenchNet, NThreads: 1
    Function Name  Total Secs    Pct
    ActFmG         2.121      8.223
    AvgMaxAct      0.1003     0.389
    AvgMaxGe       0.1012     0.3922
    DWt            5.069     19.65
    GeFmGeInc      0.3249     1.259
    InhibFmGeAct   0.08498    0.3295
    QuarterFinal   0.003773   0.01463
    SendGeDelta   14.36      55.67
    WtBalFmWt      0.1279     0.4957
    WtFmDWt        3.501     13.58
    Total         25.79
```

```
TimerReport: BenchNet, NThreads: 1
    Function Name    Total Secs    Pct
    ActFmG           2.119     7.074
    AvgMaxAct        0.1        0.3339
    AvgMaxGe        0.102     0.3407
    DWt             5.345     17.84
    GeFmGeInc        0.3348     1.118
    InhibFmGeAct     0.0842     0.2811
    QuarterFinal     0.004    0.01351
    SendGeDelta     17.93     59.87
    WtBalFmWt        0.1701     0.568
    WtFmDWt        3.763     12.56
    Total         29.96
```

* trimmed 4+ sec from SendGeDelta by avoiding range checks using sub-slices
* was very sensitive to size of Synapse struct


