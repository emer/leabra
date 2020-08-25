package pvlv

// amygdala-specific modulation parameters
type AmygModParams struct {
	DALrnThr    float32 `desc:"minimum threshold for phasic abs(da) signals to count as non-zero;  useful to screen out spurious da signals due to tiny VSPatch-to-LHb signals on t2 & t4 timesteps that can accumulate over many trials - 0.02 seems to work okay"`
	ActDeltaThr float32 `desc:"minimum threshold for delta activation to count as non-zero;  useful to screen out spurious learning due to unintended delta activity - 0.02 seems to work okay for both acquisition and extinction guys"`
	USClampAvg  float32 `desc:"averaging factor for quasi-clamping US (PV) values when sent using a SendPVAct connection to modulate net_syn values which in turn modulates actual activation values -- more graded form of clamping"`
}

// Common functionality for both BL and CEl amygdala
// More of a mixin than a complete Layer
type AmygdalaLayer struct {
	ModLayer
	AmygModParams `desc:"amygdala params"`
}

func (ly *AmygdalaLayer) ClampAvgNetin(ext, netSyn float32) float32 {
	avgGain := ly.Act.Clamp.AvgGain
	clampAvg := avgGain*ly.Act.Clamp.Gain*ext + (1.0-avgGain)*netSyn
	return clampAvg
}
