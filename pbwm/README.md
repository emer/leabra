# PBWM

[![GoDoc](https://godoc.org/github.com/emer/leabra/pbwm?status.svg)](https://godoc.org/github.com/emer/leabra/pbwm)


# Timing

## Quarter and longer

`ThalGateState.Cnt` provides key tracker of gating state:
* -1 = initialized to this value, not maintaining
* 0 = just gated – any time the thal activity exceeds the gating threshold we reset counter (re-gate)
* >= 1: maintaining – first gating goes to 1 in Quarter_Init just following the gating quarter, counts up thereafter.
* <= -1: not maintaining – when cleared, reset to -1 in Quarter_Init just following clearing quarter, counts down thereafter.


| Trial.Qtr | Phase | BG                       | PFCmnt                                          | PFCmntD                                                    | PFCout                                                 | PFCoutD                                                    | Notes                                           |
|-----------|-------|--------------------------|-------------------------------------------------|--------------------------------------------------------------|--------------------------------------------------------|--------------------------------------------------------------|-------------------------------------------------|
| 1.1       | init  |                          |                                                 |                                                              |                                                        |                                                              |                                                 |
| 1.1       | --    | GPi Go -&gt; PFCmnt.thal | input -&gt; act                                 | Ctxt = 0, no act                                       | input -&gt; act                                        | Ctxt = 0, no act                                       | cortico-cortical super only, gate mid-quarter   |
| 1.2       | --    |                          | Burst = (act &gt; thr) * thal_eff         | "                                                            | "                                                      | "                                                            | mnt super d5b gets gating signal                |
| 1.3       | init  |                          | thal &gt; 0: ThalCnt++                        | s Burst -&gt; Ctxt, Ctxt &gt; 0: ThalCnt++ |                                                        | "                                                            | mnt deep gets d5b gated "context"               |
| 1.3       | --    |                          | DeepLrn + input -&gt; act                     | Ctxt -&gt; act -&gt; s DeepLrn; out?; trc            | gets mnt deep input (opt)                              | "                                                            | mnt continues w/ no out gating, out gets primed |
| 1.4       | +     | Other Go                 | Burst = (act &gt; thr) * thal_eff         | trc d5b clamp -&gt; netin, learning                          | "                                                      | "                                                            | TRC auto-encoder learning for mnt, via deep     |
| 2.1       | init  |                          | ThalCnt &gt; 0: ThalCnt++                   | s Burst -&gt; Ctxt, Ctxt &gt; 0: ThalCnt++ |                                                        | "                                                            | mnt continues                                   |
| 2.1       | --    |                          | "                                               | "                                                            | "                                                      | "                                                            |                                                 |
| 2.2       | --    | GPi Go -&gt; PFCout.thal | "                                               | "                                                            | Burst = (act &gt; thr) * thal_eff                | "                                                            | out super d5b gets gating signal                |
| 2.3       | init  |                          | ThalCnt &lt; 0: cleared fm out, Burst = 0 | Ctxt = 0, ThalCnt = -1                               | thal &gt; 0: ThalCnt++                               | s Burst -&gt; Ctxt, Ctxt &gt; 0: ThalCnt++ | out deep gets d5b gating                        |
| 2.3       | --    |                          | input -&gt; act                                 | Ctxt = 0, no act                                       | DeepLrn + input -&gt; act                            | Ctxt -&gt; act -&gt; s DeepLrn; output; trc          | out gating takes effect, driving actual output  |
| 2.4       | +     | Other Go                 | "                                               | "                                                            | "                                                      | "                                                            | continued output driving                        |
| 3.0       | init  |                          | ThalCnt &lt; 0: ThalCnt--                   | ThalCnt &lt; 0: ThalCnt--                                | ThalCnt &gt; out_mnt: ThalCnt = -1, Burst = 0 | Ctxt = 0: ThalCnt = -1                               | out gating cleared automatically after 1 trial  |


## Cycle

### C++ version

* Gating Cyc:
    + ComputeAct:
        + GpiInvUnitSpec: detects gating, sends Thal signals to all who respond *next cycle*

* Cyc+1: 
    + ComputeAct:
        + Matrix: PatchShunt, SaveGatingThal, OutAChInhib in ApplyInhib
        + PFC:    PFCGating


