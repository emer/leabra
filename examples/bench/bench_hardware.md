# Hardware benchmarks

## MacBook Pro 16-inch, 2019: 2.4 Ghz 8-Core Intel Core i9, 64 GB 2667 Mhz DDR4 memory, Go 1.15.4

```
Size     1 thr  2 thr  4 thr
---------------------------------
SMALL:   1.25   3.31   3.51
MEDIUM:  1.59   2.26   2.07
LARGE:   9.43   7.01   5.35
HUGE:   18.6   12.9    9.66
GINORM: 23.1   17.4   13.2
```

## hpc2: Dual AMD EPYC 7532 CPUs (128 threads per node), and 256 GB of RAM each, Go 1.15.6

```
Size     1 thr  2 thr  4 thr
---------------------------------
SMALL:   1.29   4.54   5.7
MEDIUM:  1.7    4.32   4.25
LARGE:  11.2   13.4   10.3
HUGE:   22.1   18.8   13.6
GINORM: 26.6   22.6   16.9
```

## crick: Dual Intel Xeon E5-2620 V4 @ 2.10 Ghz, 64 GB RAM, Go 1.15.6

```
Size     1 thr  2 thr  4 thr
---------------------------------
SMALL:   1.91   5.2    7.18
MEDIUM:  2.28   3.62   5.19
LARGE:  12.1   14.4   12.1
HUGE:   24.0   26.5   19.2
GINORM: 30.0   33.5   24.5
```

## blanca: Dual Intel Xeon E5-2667 V2 @3.3 Ghz, 64 GB Ram, Go 1.13.4

```
Size     1 thr  2 thr  4 thr
---------------------------------
SMALL:   1.6    5.07   5.99
MEDIUM:  2.04   4.64   4.68
LARGE:  11.2   12.3    9.52
HUGE:   21.2   21.7   15.2
GINORM: 27.0   28.5   21.4
```

