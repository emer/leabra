// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package leabra

import (
	"fmt"
	"testing"

	"cogentcore.org/core/math32"
	"cogentcore.org/core/tensor"
	"github.com/emer/emergent/v2/params"
	"github.com/emer/emergent/v2/paths"
)

// Note: this test project exactly reproduces the configuration and behavior of
// C++ emergent/demo/leabra/basic_leabra_test.proj  in version 8.5.6 svn 11492

// number of distinct sets of learning parameters to test
const NLrnPars = 4

// Note: subsequent params applied after Base
var ParamSets = params.Sets{
	"Base": {
		{Sel: "Layer", Desc: "layer defaults",
			Params: params.Params{
				"Layer.Act.Gbar.L": "0.2", // was default when test created, now is 0.1
			}},
		{Sel: "Path", Desc: "for reproducibility, identical weights",
			Params: params.Params{
				"Path.WtInit.Var":        "0",
				"Path.Learn.Norm.On":     "false",
				"Path.Learn.Momentum.On": "false",
			}},
		{Sel: ".Back", Desc: "top-down back-pathways MUST have lower relative weight scale, otherwise network hallucinates",
			Params: params.Params{
				"Path.WtScale.Rel": "0.2",
			}},
	},
	"NormOn": {
		{Sel: "Path", Desc: "norm on",
			Params: params.Params{
				"Path.Learn.Norm.On": "true",
			}},
	},
	"MomentOn": {
		{Sel: "Path", Desc: "moment on",
			Params: params.Params{
				"Path.Learn.Momentum.On": "true",
			}},
	},
	"NormMomentOn": {
		{Sel: "Path", Desc: "moment on",
			Params: params.Params{
				"Path.Learn.Momentum.On": "true",
				"Path.Learn.Norm.On":     "true",
			}},
	},
}

func MakeTestNet(t *testing.T) *Network {
	testNet := NewNetwork("TestNet")
	inLay := testNet.AddLayer("Input", []int{4, 1}, InputLayer)
	hidLay := testNet.AddLayer("Hidden", []int{4, 1}, SuperLayer)
	outLay := testNet.AddLayer("Output", []int{4, 1}, TargetLayer)

	testNet.ConnectLayers(inLay, hidLay, paths.NewOneToOne(), ForwardPath)
	testNet.ConnectLayers(hidLay, outLay, paths.NewOneToOne(), ForwardPath)
	testNet.ConnectLayers(outLay, hidLay, paths.NewOneToOne(), BackPath)

	testNet.Defaults()
	testNet.ApplyParams(ParamSets["Base"], false) // false) // true) // no msg
	testNet.Build()
	testNet.InitWeights()
	testNet.AlphaCycInit(true) // get GScale

	// var buf bytes.Buffer
	// TestNet.WriteWtsJSON(&buf)
	// wb := buf.Bytes()
	// // fmt.Printf("TestNet Weights:\n\n%v\n", string(wb))
	//
	// fp, err := os.Create("testdata/testnet.wts")
	// defer fp.Close()
	// if err != nil {
	// 	t.Error(err)
	// }
	// fp.Write(wb)
	return testNet
}

func TestSynValues(t *testing.T) {
	testNet := MakeTestNet(t)
	testNet.InitWeights()
	hidLay := testNet.LayerByName("Hidden")
	fmIni, _ := hidLay.RecvPathBySendName("Input")
	fmIn := fmIni.(*Path)

	bfWt := fmIn.SynValue("Wt", 1, 1)
	if math32.IsNaN(bfWt) {
		t.Errorf("Wt syn var not found")
	}
	bfLWt := fmIn.SynValue("LWt", 1, 1)

	fmIn.SetSynValue("Wt", 1, 1, .15)

	afWt := fmIn.SynValue("Wt", 1, 1)
	afLWt := fmIn.SynValue("LWt", 1, 1)

	CmprFloats([]float32{bfWt, bfLWt, afWt, afLWt}, []float32{0.5, 0.5, 0.15, 0.42822415}, "syn val setting test", t)

	// fmt.Printf("SynValues: before wt: %v, lwt: %v  after wt: %v, lwt: %v\n", bfWt, bfLWt, afWt, afLWt)
}

func MakeInPats(t *testing.T) *tensor.Float32 {
	inPats := tensor.NewFloat32([]int{4, 4, 1}, "pat", "Y", "X")
	for pi := 0; pi < 4; pi++ {
		inPats.Set([]int{pi, pi, 0}, 1)
	}
	return inPats
}

func CmprFloats(got, trg []float32, msg string, t *testing.T) {
	for i := range got {
		dif := math32.Abs(got[i] - trg[i])
		if dif > difTol { // allow for small numerical diffs
			t.Errorf("%v err: got: %v, trg: %v, dif: %v\n", msg, got[i], trg[i], dif)
		}
	}
}

func TestNetAct(t *testing.T) {
	inPats := MakeInPats(t)
	testNet := MakeTestNet(t)
	testNet.InitWeights()
	testNet.InitExt()

	inLay := testNet.LayerByName("Input")
	hidLay := testNet.LayerByName("Hidden")
	outLay := testNet.LayerByName("Output")

	ctx := NewContext()

	printCycs := false
	printQtrs := false

	qtr0HidActs := []float32{0.9415894, 2.4012093e-33, 2.4012093e-33, 2.4012093e-33}
	qtr0HidGes := []float32{0.4715216, 0, 0, 0}
	qtr0HidGis := []float32{0.45561486, 0.45561486, 0.45561486, 0.45561486}
	qtr0OutActs := []float32{0.9406784, 2.4021936e-33, 2.4021936e-33, 2.4021936e-33}
	qtr0OutGes := []float32{0.46987593, 0, 0, 0}
	qtr0OutGis := []float32{0.45427382, 0.45427382, 0.45427382, 0.45427382}

	qtr3HidActs := []float32{0.9431544, 4e-45, 4e-45, 4e-45}
	qtr3HidGes := []float32{0.47499993, 0, 0, 0}
	qtr3HidGis := []float32{0.45816946, 0.45816946, 0.45816946, 0.45816946}
	qtr3OutActs := []float32{0.95, 0, 0, 0}
	qtr3OutGes := []float32{0.4699372, 0, 0, 0}
	qtr3OutGis := []float32{0.4589717, 0.4589717, 0.4589717, 0.4589717}

	inActs := []float32{}
	hidActs := []float32{}
	hidGes := []float32{}
	hidGis := []float32{}
	outActs := []float32{}
	outGes := []float32{}
	outGis := []float32{}

	for pi := 0; pi < 4; pi++ {
		inpat := inPats.SubSpace([]int{pi})
		inLay.ApplyExt(inpat)
		outLay.ApplyExt(inpat)

		testNet.AlphaCycInit(true)
		ctx.AlphaCycStart()
		for qtr := 0; qtr < 4; qtr++ {
			for cyc := 0; cyc < ctx.CycPerQtr; cyc++ {
				testNet.Cycle(ctx)
				ctx.CycleInc()

				if printCycs {
					inLay.UnitValues(&inActs, "Act", 0)
					hidLay.UnitValues(&hidActs, "Act", 0)
					hidLay.UnitValues(&hidGes, "Ge", 0)
					hidLay.UnitValues(&hidGis, "Gi", 0)
					outLay.UnitValues(&outActs, "Act", 0)
					outLay.UnitValues(&outGes, "Ge", 0)
					outLay.UnitValues(&outGis, "Gi", 0)
					fmt.Printf("pat: %v qtr: %v cyc: %v\nin acts: %v\nhid acts: %v ges: %v gis: %v\nout acts: %v ges: %v gis: %v\n", pi, qtr, cyc, inActs, hidActs, hidGes, hidGis, outActs, outGes, outGis)
				}
			}
			testNet.QuarterFinal(ctx)
			ctx.QuarterInc()

			if printCycs && printQtrs {
				fmt.Printf("=============================\n")
			}

			inLay.UnitValues(&inActs, "Act", 0)
			hidLay.UnitValues(&hidActs, "Act", 0)
			hidLay.UnitValues(&hidGes, "Ge", 0)
			hidLay.UnitValues(&hidGis, "Gi", 0)
			outLay.UnitValues(&outActs, "Act", 0)
			outLay.UnitValues(&outGes, "Ge", 0)
			outLay.UnitValues(&outGis, "Gi", 0)

			if printQtrs {
				fmt.Printf("pat: %v qtr: %v cyc: %v\nin acts: %v\nhid acts: %v ges: %v gis: %v\nout acts: %v ges: %v gis: %v\n", pi, qtr, ctx.Cycle, inActs, hidActs, hidGes, hidGis, outActs, outGes, outGis)
			}

			if printCycs && printQtrs {
				fmt.Printf("=============================\n")
			}

			if pi == 0 && qtr == 0 {
				CmprFloats(hidActs, qtr0HidActs, "qtr 0 hidActs", t)
				CmprFloats(hidGes, qtr0HidGes, "qtr 0 hidGes", t)
				CmprFloats(hidGis, qtr0HidGis, "qtr 0 hidGis", t)
				CmprFloats(outActs, qtr0OutActs, "qtr 0 outActs", t)
				CmprFloats(outGes, qtr0OutGes, "qtr 0 outGes", t)
				CmprFloats(outGis, qtr0OutGis, "qtr 0 outGis", t)
			}
			if pi == 0 && qtr == 3 {
				CmprFloats(hidActs, qtr3HidActs, "qtr 3 hidActs", t)
				CmprFloats(hidGes, qtr3HidGes, "qtr 3 hidGes", t)
				CmprFloats(hidGis, qtr3HidGis, "qtr 3 hidGis", t)
				CmprFloats(outActs, qtr3OutActs, "qtr 3 outActs", t)
				CmprFloats(outGes, qtr3OutGes, "qtr 3 outGes", t)
				CmprFloats(outGis, qtr3OutGis, "qtr 3 outGis", t)
			}
		}

		if printQtrs {
			fmt.Printf("=============================\n")
		}
	}
}

func TestNetLearn(t *testing.T) {
	inPats := MakeInPats(t)
	testNet := MakeTestNet(t)
	inLay := testNet.LayerByName("Input")
	hidLay := testNet.LayerByName("Hidden")
	outLay := testNet.LayerByName("Output")

	printCycs := false
	printQtrs := false

	qtr0HidAvgS := []float32{0.9422413, 6.034972e-08, 6.034972e-08, 6.034972e-08}
	qtr0HidAvgM := []float32{0.8162388, 0.013628835, 0.013628835, 0.013628835}
	qtr0OutAvgS := []float32{0.93967456, 6.034972e-08, 6.034972e-08, 6.034972e-08}
	qtr0OutAvgM := []float32{0.7438192, 0.013628835, 0.013628835, 0.013628835}

	qtr3HidAvgS := []float32{0.94315434, 6.0347804e-30, 6.0347804e-30, 6.0347804e-30}
	qtr3HidAvgM := []float32{0.94308215, 5.042516e-06, 5.042516e-06, 5.042516e-06}
	qtr3OutAvgS := []float32{0.9499999, 6.0347804e-30, 6.0347804e-30, 6.0347804e-30}
	qtr3OutAvgM := []float32{0.9492211, 5.042516e-06, 5.042516e-06, 5.042516e-06}

	trl0HidAvgL := []float32{0.3975, 0.3975, 0.3975, 0.3975}
	trl1HidAvgL := []float32{0.5935205, 0.35775128, 0.35775128, 0.35775128}
	trl2HidAvgL := []float32{0.5341791, 0.55774546, 0.32197616, 0.32197616}
	trl3HidAvgL := []float32{0.48076117, 0.50198156, 0.5255478, 0.28977853}

	trl1HidAvgLLrn := []float32{0.0008553083, 0.00034286897, 0.00034286897, 0.00034286897}
	trl2HidAvgLLrn := []float32{0.000726331, 0.00077755196, 0.00026511253, 0.00026511253}
	trl3HidAvgLLrn := []float32{0.00061022834, 0.0006563504, 0.0007075711, 0.00019513167}

	// these are organized by pattern within and then by test iteration (params) outer
	hidDwts := []float32{3.376007e-06, 1.1105859e-05, 9.811188e-06, 8.4557105e-06,
		0.00050640106, 0.0016658787, 0.0014716781, 0.0012683566,
		3.376007e-07, 1.1105858e-06, 9.811188e-07, 8.4557104e-07,
		5.0640105e-05, 0.00016658788, 0.00014716781, 0.00012683566}
	outDwts := []float32{2.8908253e-05, 2.9251574e-05, 2.9251574e-05, 2.9251574e-05,
		0.004336238, 0.0043877363, 0.0043877363, 0.0043877363,
		2.8908253e-06, 2.9251576e-06, 2.9251576e-06, 2.9251576e-06,
		0.0004336238, 0.00043877363, 0.00043877363, 0.00043877363}
	hidNorms := []float32{0, 0, 0, 0, 8.440018e-05, 0.00027764647, 0.0002452797, 0.00021139276,
		0, 0, 0, 0, 8.440018e-05, 0.00027764647, 0.0002452797, 0.00021139276}
	outNorms := []float32{0, 0, 0, 0, 0.0007227063, 0.0007312894, 0.0007312894, 0.0007312894,
		0, 0, 0, 0, 0.0007227063, 0.0007312894, 0.0007312894, 0.0007312894}
	hidMoments := []float32{0, 0, 0, 0, 0, 0, 0, 0,
		8.440018e-05, 0.00027764647, 0.0002452797, 0.00021139276,
		8.440018e-05, 0.00027764647, 0.0002452797, 0.00021139276}
	outMoments := []float32{0, 0, 0, 0, 0, 0, 0, 0,
		0.0007227063, 0.0007312894, 0.0007312894, 0.0007312894,
		0.0007227063, 0.0007312894, 0.0007312894, 0.0007312894}
	hidWts := []float32{0.50001, 0.50003326, 0.5000293, 0.5000254,
		0.50151914, 0.5049973, 0.5044148, 0.5038051,
		0.5000011, 0.5000032, 0.50000286, 0.5000025,
		0.500152, 0.5004996, 0.5004417, 0.5003805}
	outWts := []float32{0.50008655, 0.5000876, 0.5000876, 0.5000876,
		0.51300585, 0.5131602, 0.5131602, 0.5131602,
		0.5000086, 0.50000894, 0.50000894, 0.50000894,
		0.5013011, 0.5013164, 0.5013164, 0.5013164}

	hiddwt := make([]float32, 4*NLrnPars)
	outdwt := make([]float32, 4*NLrnPars)
	hidwt := make([]float32, 4*NLrnPars)
	outwt := make([]float32, 4*NLrnPars)
	hidnorm := make([]float32, 4*NLrnPars)
	outnorm := make([]float32, 4*NLrnPars)
	hidmoment := make([]float32, 4*NLrnPars)
	outmoment := make([]float32, 4*NLrnPars)

	hidAct := []float32{}
	hidGes := []float32{}
	hidGis := []float32{}
	hidAvgSS := []float32{}
	hidAvgS := []float32{}
	hidAvgM := []float32{}
	outAvgS := []float32{}
	outAvgM := []float32{}
	hidAvgL := []float32{}
	hidAvgLLrn := []float32{}
	outAvgL := []float32{}
	outAvgLLrn := []float32{}

	for ti := 0; ti < NLrnPars; ti++ {
		testNet.Defaults()
		testNet.ApplyParams(ParamSets["Base"], false) // always apply base
		// testNet.ApplyParams(ParamSets[ti].Sheets["Network"], false) // then specific
		testNet.InitWeights()
		testNet.InitExt()

		ctx := NewContext()

		for pi := 0; pi < 4; pi++ {
			inpat := inPats.SubSpace([]int{pi})
			inLay.ApplyExt(inpat)
			outLay.ApplyExt(inpat)

			testNet.AlphaCycInit(true)
			ctx.AlphaCycStart()
			for qtr := 0; qtr < 4; qtr++ {
				for cyc := 0; cyc < ctx.CycPerQtr; cyc++ {
					testNet.Cycle(ctx)
					ctx.CycleInc()

					hidLay.UnitValues(&hidAct, "Act", 0)
					hidLay.UnitValues(&hidGes, "Ge", 0)
					hidLay.UnitValues(&hidGis, "Gi", 0)
					hidLay.UnitValues(&hidAvgSS, "AvgSS", 0)
					hidLay.UnitValues(&hidAvgS, "AvgS", 0)
					hidLay.UnitValues(&hidAvgM, "AvgM", 0)

					outLay.UnitValues(&outAvgS, "AvgS", 0)
					outLay.UnitValues(&outAvgM, "AvgM", 0)

					if printCycs {
						fmt.Printf("pat: %v qtr: %v cyc: %v\nhid act: %v ges: %v gis: %v\nhid avgss: %v avgs: %v avgm: %v\nout avgs: %v avgm: %v\n", pi, qtr, ctx.Cycle, hidAct, hidGes, hidGis, hidAvgSS, hidAvgS, hidAvgM, outAvgS, outAvgM)
					}

				}
				testNet.QuarterFinal(ctx)
				ctx.QuarterInc()

				hidLay.UnitValues(&hidAvgS, "AvgS", 0)
				hidLay.UnitValues(&hidAvgM, "AvgM", 0)

				outLay.UnitValues(&outAvgS, "AvgS", 0)
				outLay.UnitValues(&outAvgM, "AvgM", 0)

				if printQtrs {
					fmt.Printf("pat: %v qtr: %v cyc: %v\nhid avgs: %v avgm: %v\nout avgs: %v avgm: %v\n", pi, qtr, ctx.Cycle, hidAvgS, hidAvgM, outAvgS, outAvgM)
				}

				if pi == 0 && qtr == 0 {
					CmprFloats(hidAvgS, qtr0HidAvgS, "qtr 0 hidAvgS", t)
					CmprFloats(hidAvgM, qtr0HidAvgM, "qtr 0 hidAvgM", t)
					CmprFloats(outAvgS, qtr0OutAvgS, "qtr 0 outAvgS", t)
					CmprFloats(outAvgM, qtr0OutAvgM, "qtr 0 outAvgM", t)
				}
				if pi == 0 && qtr == 3 {
					CmprFloats(hidAvgS, qtr3HidAvgS, "qtr 3 hidAvgS", t)
					CmprFloats(hidAvgM, qtr3HidAvgM, "qtr 3 hidAvgM", t)
					CmprFloats(outAvgS, qtr3OutAvgS, "qtr 3 outAvgS", t)
					CmprFloats(outAvgM, qtr3OutAvgM, "qtr 3 outAvgM", t)
				}
			}

			if printQtrs {
				fmt.Printf("=============================\n")
			}

			hidLay.UnitValues(&hidAvgL, "AvgL", 0)
			hidLay.UnitValues(&hidAvgLLrn, "AvgLLrn", 0)
			outLay.UnitValues(&outAvgL, "AvgL", 0)
			outLay.UnitValues(&outAvgLLrn, "AvgLLrn", 0)
			_ = outAvgL
			_ = outAvgLLrn

			// fmt.Printf("hid cosdif stats: %v\nhid avgl:   %v\nhid avgllrn: %v\n", hidLay.CosDiff, hidAvgL, hidAvgLLrn)
			// fmt.Printf("out cosdif stats: %v\nout avgl:   %v\nout avgllrn: %v\n", outLay.CosDiff, outAvgL, outAvgLLrn)

			testNet.DWt()

			didx := ti*4 + pi

			hiddwt[didx] = hidLay.RecvPaths[0].SynValue("DWt", pi, pi)
			outdwt[didx] = outLay.RecvPaths[0].SynValue("DWt", pi, pi)
			hidnorm[didx] = hidLay.RecvPaths[0].SynValue("Norm", pi, pi)
			outnorm[didx] = outLay.RecvPaths[0].SynValue("Norm", pi, pi)
			hidmoment[didx] = hidLay.RecvPaths[0].SynValue("Moment", pi, pi)
			outmoment[didx] = outLay.RecvPaths[0].SynValue("Moment", pi, pi)

			testNet.WtFromDWt()

			hidwt[didx] = hidLay.RecvPaths[0].SynValue("Wt", pi, pi)
			outwt[didx] = outLay.RecvPaths[0].SynValue("Wt", pi, pi)

			switch pi {
			case 0:
				CmprFloats(hidAvgL, trl0HidAvgL, "trl 0 hidAvgL", t)
			case 1:
				CmprFloats(hidAvgL, trl1HidAvgL, "trl 1 hidAvgL", t)
				CmprFloats(hidAvgLLrn, trl1HidAvgLLrn, "trl 1 hidAvgLLrn", t)
			case 2:
				CmprFloats(hidAvgL, trl2HidAvgL, "trl 2 hidAvgL", t)
				CmprFloats(hidAvgLLrn, trl2HidAvgLLrn, "trl 2 hidAvgLLrn", t)
			case 3:
				CmprFloats(hidAvgL, trl3HidAvgL, "trl 3 hidAvgL", t)
				CmprFloats(hidAvgLLrn, trl3HidAvgLLrn, "trl 3 hidAvgLLrn", t)
			}

		}
	}

	//	fmt.Printf("hid dwt: %v\nout dwt: %v\nhid norm: %v\n hid moment: %v\nout norm: %v\nout moment: %v\nhid wt: %v\nout wt: %v\n", hiddwt, outdwt, hidnorm, hidmoment, outnorm, outmoment, hidwt, outwt)

	CmprFloats(hiddwt, hidDwts, "hid DWt", t)
	CmprFloats(outdwt, outDwts, "out DWt", t)
	CmprFloats(hidnorm, hidNorms, "hid Norm", t)
	CmprFloats(outnorm, outNorms, "out Norm", t)
	CmprFloats(hidmoment, hidMoments, "hid Moment", t)
	CmprFloats(outmoment, outMoments, "out Moment", t)
	CmprFloats(hidwt, hidWts, "hid Wt", t)
	CmprFloats(outwt, outWts, "out Wt", t)

	// var buf bytes.Buffer
	// TestNet.WriteWtsJSON(&buf)
	// wb := buf.Bytes()
	// fmt.Printf("TestNet Trained Weights:\n\n%v\n", string(wb))

	// fp, err := os.Create("testdata/testnet_train.wts")
	// defer fp.Close()
	// if err != nil {
	// 	t.Error(err)
	// }
	// fp.Write(wb)
}

func TestInhibAct(t *testing.T) {
	inPats := MakeInPats(t)
	inhibNet := NewNetwork("InhibNet")

	inLay := inhibNet.AddLayer("Input", []int{4, 1}, InputLayer)
	hidLay := inhibNet.AddLayer("Hidden", []int{4, 1}, SuperLayer)
	outLay := inhibNet.AddLayer("Output", []int{4, 1}, TargetLayer)

	inhibNet.ConnectLayers(inLay, hidLay, paths.NewOneToOne(), ForwardPath)
	inhibNet.ConnectLayers(inLay, hidLay, paths.NewOneToOne(), InhibPath)
	inhibNet.ConnectLayers(hidLay, outLay, paths.NewOneToOne(), ForwardPath)
	inhibNet.ConnectLayers(outLay, hidLay, paths.NewOneToOne(), BackPath)

	inhibNet.Defaults()
	inhibNet.ApplyParams(ParamSets["Base"], false) // true) // no msg
	inhibNet.Build()
	inhibNet.InitWeights()
	inhibNet.AlphaCycInit(true) // get GScale

	inhibNet.InitWeights()
	inhibNet.InitExt()

	ctx := NewContext()

	printCycs := false
	printQtrs := false

	qtr0HidActs := []float32{0.49207208, 2.4012093e-33, 2.4012093e-33, 2.4012093e-33}
	qtr0HidGes := []float32{0.44997975, 0, 0, 0}
	qtr0HidGis := []float32{0.71892214, 0.24392211, 0.24392211, 0.24392211}
	qtr0OutActs := []float32{0.648718, 2.4021936e-33, 2.4021936e-33, 2.4021936e-33}
	qtr0OutGes := []float32{0.24562117, 0, 0, 0}
	qtr0OutGis := []float32{0.29192635, 0.29192635, 0.29192635, 0.29192635}

	qtr3HidActs := []float32{0.5632278, 4e-45, 4e-45, 4e-45}
	qtr3HidGes := []float32{0.475, 0, 0, 0}
	qtr3HidGis := []float32{0.7622025, 0.28720248, 0.28720248, 0.28720248}
	qtr3OutActs := []float32{0.95, 0, 0, 0}
	qtr3OutGes := []float32{0.2802849, 0, 0, 0}
	qtr3OutGis := []float32{0.42749998, 0.42749998, 0.42749998, 0.42749998}

	inActs := []float32{}
	hidActs := []float32{}
	hidGes := []float32{}
	hidGis := []float32{}
	outActs := []float32{}
	outGes := []float32{}
	outGis := []float32{}

	for pi := 0; pi < 4; pi++ {
		inpat := inPats.SubSpace([]int{pi})
		inLay.ApplyExt(inpat)
		outLay.ApplyExt(inpat)

		inhibNet.AlphaCycInit(true)
		ctx.AlphaCycStart()
		for qtr := 0; qtr < 4; qtr++ {
			for cyc := 0; cyc < ctx.CycPerQtr; cyc++ {
				inhibNet.Cycle(ctx)
				ctx.CycleInc()

				if printCycs {
					inLay.UnitValues(&inActs, "Act", 0)
					hidLay.UnitValues(&hidActs, "Act", 0)
					hidLay.UnitValues(&hidGes, "Ge", 0)
					hidLay.UnitValues(&hidGis, "Gi", 0)
					outLay.UnitValues(&outActs, "Act", 0)
					outLay.UnitValues(&outGes, "Ge", 0)
					outLay.UnitValues(&outGis, "Gi", 0)
					fmt.Printf("pat: %v qtr: %v cyc: %v\nin acts: %v\nhid acts: %v ges: %v gis: %v\nout acts: %v ges: %v gis: %v\n", pi, qtr, cyc, inActs, hidActs, hidGes, hidGis, outActs, outGes, outGis)
				}
			}
			inhibNet.QuarterFinal(ctx)
			ctx.QuarterInc()

			if printCycs && printQtrs {
				fmt.Printf("=============================\n")
			}

			inLay.UnitValues(&inActs, "Act", 0)
			hidLay.UnitValues(&hidActs, "Act", 0)
			hidLay.UnitValues(&hidGes, "Ge", 0)
			hidLay.UnitValues(&hidGis, "Gi", 0)
			outLay.UnitValues(&outActs, "Act", 0)
			outLay.UnitValues(&outGes, "Ge", 0)
			outLay.UnitValues(&outGis, "Gi", 0)

			if printQtrs {
				fmt.Printf("pat: %v qtr: %v cyc: %v\nin acts: %v\nhid acts: %v ges: %v gis: %v\nout acts: %v ges: %v gis: %v\n", pi, qtr, ctx.Cycle, inActs, hidActs, hidGes, hidGis, outActs, outGes, outGis)
			}

			if printCycs && printQtrs {
				fmt.Printf("=============================\n")
			}

			if pi == 0 && qtr == 0 {
				CmprFloats(hidActs, qtr0HidActs, "qtr 0 hidActs", t)
				CmprFloats(hidGes, qtr0HidGes, "qtr 0 hidGes", t)
				CmprFloats(hidGis, qtr0HidGis, "qtr 0 hidGis", t)
				CmprFloats(outActs, qtr0OutActs, "qtr 0 outActs", t)
				CmprFloats(outGes, qtr0OutGes, "qtr 0 outGes", t)
				CmprFloats(outGis, qtr0OutGis, "qtr 0 outGis", t)
			}
			if pi == 0 && qtr == 3 {
				CmprFloats(hidActs, qtr3HidActs, "qtr 3 hidActs", t)
				CmprFloats(hidGes, qtr3HidGes, "qtr 3 hidGes", t)
				CmprFloats(hidGis, qtr3HidGis, "qtr 3 hidGis", t)
				CmprFloats(outActs, qtr3OutActs, "qtr 3 outActs", t)
				CmprFloats(outGes, qtr3OutGes, "qtr 3 outGes", t)
				CmprFloats(outGis, qtr3OutGis, "qtr 3 outGis", t)
			}
		}

		if printQtrs {
			fmt.Printf("=============================\n")
		}
	}
}
