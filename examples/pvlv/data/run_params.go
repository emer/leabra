// Copyright (c) 2020, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package data

// A sequence of runs (each step is a ConditionParams object)
type RunParams struct {

	// Name of the sequence
	Nm string `desc:"Name of the sequence"`

	// Description
	Desc string `desc:"Description"`

	// name of condition 1
	Cond1Nm string `desc:"name of condition 1"`

	// name of condition 2
	Cond2Nm string `desc:"name of condition 2"`

	// name of condition 3
	Cond3Nm string `desc:"name of condition 3"`

	// name of condition 4
	Cond4Nm string `desc:"name of condition 4"`

	// name of condition 5
	Cond5Nm string `desc:"name of condition 5"`
}
type RunParamsMap map[string]RunParams

func AllRunParams() RunParamsMap {
	seqs := map[string]RunParams{
		"RunMaster": {
			Nm:      "RunMaster",
			Desc:    "",
			Cond1Nm: "PosAcq_B50",
			Cond2Nm: "NullStep",
			Cond3Nm: "NullStep",
			Cond4Nm: "NullStep",
			Cond5Nm: "NullStep",
		},
		"USDebug": {
			Nm:      "USDebug",
			Desc:    "",
			Cond1Nm: "USDebug",
			Cond2Nm: "NullStep",
			Cond3Nm: "NullStep",
			Cond4Nm: "NullStep",
			Cond5Nm: "NullStep",
		},
		"US0": {
			Nm:      "US0",
			Desc:    "",
			Cond1Nm: "US0",
			Cond2Nm: "NullStep",
			Cond3Nm: "NullStep",
			Cond4Nm: "NullStep",
			Cond5Nm: "NullStep",
		},
		"PosAcq_A50": {
			Nm:      "PosAcq_A50",
			Desc:    "",
			Cond1Nm: "PosAcq_A50",
			Cond2Nm: "NullStep",
			Cond3Nm: "NullStep",
			Cond4Nm: "NullStep",
			Cond5Nm: "NullStep",
		},
		"PosAcq_B50Ext": {
			Nm:      "PosAcq_B50Ext",
			Desc:    "",
			Cond1Nm: "PosAcq_B50",
			Cond2Nm: "PosExtinct",
			Cond3Nm: "NullStep",
			Cond4Nm: "NullStep",
			Cond5Nm: "NullStep",
		},
		"PosAcq_B50ExtAcq": {
			Nm:      "PosAcq_B50ExtAcq",
			Desc:    "Full cycle: acq, ext, acq",
			Cond1Nm: "PosAcq_B50",
			Cond2Nm: "PosExtinct",
			Cond3Nm: "PosAcq_B50Cont",
			Cond4Nm: "NullStep",
			Cond5Nm: "NullStep",
		},
		"PosAcq_B100Ext": {
			Nm:      "PosAcq_B100Ext",
			Desc:    "",
			Cond1Nm: "PosAcq_B100",
			Cond2Nm: "PosExtinct",
			Cond3Nm: "NullStep",
			Cond4Nm: "NullStep",
			Cond5Nm: "NullStep",
		},
		"PosAcq": {
			Nm:      "PosAcq",
			Desc:    "",
			Cond1Nm: "PosAcq_B50",
			Cond2Nm: "NullStep",
			Cond3Nm: "NullStep",
			Cond4Nm: "NullStep",
			Cond5Nm: "NullStep",
		},
		"PosExt": {
			Nm:      "PosExt",
			Desc:    "",
			Cond1Nm: "PosAcq_B50",
			Cond2Nm: "PosExtinct",
			Cond3Nm: "NullStep",
			Cond4Nm: "NullStep",
			Cond5Nm: "NullStep",
		},
		"PosAcq_B25": {
			Nm:      "PosAcq_B25",
			Desc:    "",
			Cond1Nm: "PosAcq_B25",
			Cond2Nm: "NullStep",
			Cond3Nm: "NullStep",
			Cond4Nm: "NullStep",
			Cond5Nm: "NullStep",
		},
		"NegAcq": {
			Nm:      "NegAcq",
			Desc:    "",
			Cond1Nm: "NegAcq",
			Cond2Nm: "NullStep",
			Cond3Nm: "NullStep",
			Cond4Nm: "NullStep",
			Cond5Nm: "NullStep",
		},
		"NegAcqMag": {
			Nm:      "NegAcqMag",
			Desc:    "",
			Cond1Nm: "NegAcqMag",
			Cond2Nm: "NullStep",
			Cond3Nm: "NullStep",
			Cond4Nm: "NullStep",
			Cond5Nm: "NullStep",
		},
		"PosAcqMag": {
			Nm:      "PosAcqMag",
			Desc:    "",
			Cond1Nm: "PosAcqMag",
			Cond2Nm: "NullStep",
			Cond3Nm: "NullStep",
			Cond4Nm: "NullStep",
			Cond5Nm: "NullStep",
		},
		"NegAcqExt": {
			Nm:      "NegAcqExt",
			Desc:    "",
			Cond1Nm: "NegAcq",
			Cond2Nm: "NegExtinct",
			Cond3Nm: "NullStep",
			Cond4Nm: "NullStep",
			Cond5Nm: "NullStep",
		},
		"PosCondInhib": {
			Nm:      "PosCondInhib",
			Desc:    "",
			Cond1Nm: "PosAcq_contextA",
			Cond2Nm: "PosCondInhib",
			Cond3Nm: "PosCondInhib_test",
			Cond4Nm: "NullStep",
			Cond5Nm: "NullStep",
		},
		"PosSecondOrderCond": {
			Nm:      "PosSecondOrderCond",
			Desc:    "",
			Cond1Nm: "PosAcqPreSecondOrder",
			Cond2Nm: "PosSecondOrderCond",
			Cond3Nm: "NullStep",
			Cond4Nm: "NullStep",
			Cond5Nm: "NullStep",
		},
		"PosBlocking": {
			Nm:      "PosBlocking",
			Desc:    "",
			Cond1Nm: "PosBlocking_A_training",
			Cond2Nm: "PosBlocking",
			Cond3Nm: "PosBlocking_test",
			Cond4Nm: "NullStep",
			Cond5Nm: "NullStep",
		},
		"PosBlocking2": {
			Nm:      "PosBlocking2",
			Desc:    "",
			Cond1Nm: "PosBlocking_A_training",
			Cond2Nm: "PosBlocking",
			Cond3Nm: "PosBlocking2_test",
			Cond4Nm: "NullStep",
			Cond5Nm: "NullStep",
		},
		"NegCondInhib": {
			Nm:      "NegCondInhib",
			Desc:    "",
			Cond1Nm: "NegAcq",
			Cond2Nm: "NegCondInh",
			Cond3Nm: "NegCondInh_test",
			Cond4Nm: "NullStep",
			Cond5Nm: "NullStep",
		},
		"AbaRenewal": {
			Nm:      "AbaRenewal",
			Desc:    "",
			Cond1Nm: "PosAcq_contextA",
			Cond2Nm: "PosExtinct_contextB",
			Cond3Nm: "PosRenewal_contextA",
			Cond4Nm: "NullStep",
			Cond5Nm: "NullStep",
		},
		"NegBlocking": {
			Nm:      "NegBlocking",
			Desc:    "",
			Cond1Nm: "NegBlocking_E_training",
			Cond2Nm: "NegBlocking",
			Cond3Nm: "NegBlocking_test",
			Cond4Nm: "NullStep",
			Cond5Nm: "NullStep",
		},
		"PosSum_test": {
			Nm:      "PosSum_test",
			Desc:    "",
			Cond1Nm: "PosSumAcq",
			Cond2Nm: "PosSumCondInhib",
			Cond3Nm: "PosSum_test",
			Cond4Nm: "NullStep",
			Cond5Nm: "NullStep",
		},
		"NegSum_test": {
			Nm:      "NegSum_test",
			Desc:    "",
			Cond1Nm: "NegSumAcq",
			Cond2Nm: "NegSumCondInhib",
			Cond3Nm: "NegSum_test",
			Cond4Nm: "NullStep",
			Cond5Nm: "NullStep",
		},
		"UnblockingValue": {
			Nm:      "UnblockingValue",
			Desc:    "",
			Cond1Nm: "Unblocking_train",
			Cond2Nm: "UnblockingValue",
			Cond3Nm: "UnblockingValue_test",
			Cond4Nm: "NullStep",
			Cond5Nm: "NullStep",
		},
		"UnblockingIdentity": {
			Nm:      "UnblockingIdentity",
			Desc:    "",
			Cond1Nm: "Unblocking_trainUS",
			Cond2Nm: "UnblockingIdentity",
			Cond3Nm: "UnblockingIdentity_test",
			Cond4Nm: "NullStep",
			Cond5Nm: "NullStep",
		},
		"Overexpect": {
			Nm:      "Overexpect",
			Desc:    "",
			Cond1Nm: "Overexpect_train",
			Cond2Nm: "OverexpectCompound",
			Cond3Nm: "Overexpect_test",
			Cond4Nm: "NullStep",
			Cond5Nm: "NullStep",
		},
		"PosMagChange": {
			Nm:      "PosMagChange",
			Desc:    "",
			Cond1Nm: "PosAcqMag",
			Cond2Nm: "PosAcqMagChange",
			Cond3Nm: "Overexpect_test",
			Cond4Nm: "NullStep",
			Cond5Nm: "NullStep",
		},
		"NegMagChange": {
			Nm:      "NegMagChange",
			Desc:    "",
			Cond1Nm: "NegAcqMag",
			Cond2Nm: "NegAcqMagChange",
			Cond3Nm: "NullStep",
			Cond4Nm: "NullStep",
			Cond5Nm: "NullStep",
		},
		"CondExp": {
			Nm:      "CondExp",
			Desc:    "",
			Cond1Nm: "CondExp",
			Cond2Nm: "NullStep",
			Cond3Nm: "NullStep",
			Cond4Nm: "NullStep",
			Cond5Nm: "NullStep",
		},
		"PainExp": {
			Nm:      "PainExp",
			Desc:    "",
			Cond1Nm: "PainExp",
			Cond2Nm: "NullStep",
			Cond3Nm: "NullStep",
			Cond4Nm: "NullStep",
			Cond5Nm: "NullStep",
		},
		"PosNeg": {
			Nm:      "PosNeg",
			Desc:    "",
			Cond1Nm: "PosOrNegAcq",
			Cond2Nm: "NullStep",
			Cond3Nm: "NullStep",
			Cond4Nm: "NullStep",
			Cond5Nm: "NullStep",
		},
		"PosAcqEarlyUSTest": {
			Nm:      "PosAcqEarlyUSTest",
			Desc:    "",
			Cond1Nm: "PosAcq_B50",
			Cond2Nm: "PosAcqEarlyUS_test",
			Cond3Nm: "NullStep",
			Cond4Nm: "NullStep",
			Cond5Nm: "NullStep",
		},
		"AutomatedTesting": {
			Nm:      "AutomatedTesting",
			Desc:    "This paramset is just for naming purposes",
			Cond1Nm: "NullStep",
			Cond2Nm: "NullStep",
			Cond3Nm: "NullStep",
			Cond4Nm: "NullStep",
			Cond5Nm: "NullStep",
		},
		"PosOrNegAcq": {
			Nm:      "PosOrNegAcq",
			Desc:    "",
			Cond1Nm: "PosOrNegAcq",
			Cond2Nm: "NullStep",
			Cond3Nm: "NullStep",
			Cond4Nm: "NullStep",
			Cond5Nm: "NullStep",
		},
		"PosCondInhib_test": {
			Nm:      "PosCondInhib_test",
			Desc:    "For debugging",
			Cond1Nm: "PosCondInhib_test",
			Cond2Nm: "NullStep",
			Cond3Nm: "NullStep",
			Cond4Nm: "NullStep",
			Cond5Nm: "NullStep",
		},
	}

	return seqs
}
