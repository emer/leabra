// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Package hip provides special hippocampus algorithms for implementing the Theta-phase
hippocampus model from Ketz, Morkonda, & O'Reilly (2013).

timing of ThetaPhase dynamics -- based on quarter structure:

Q1:   ECin -> CA1 -> ECout (CA3 -> CA1 off)  : ActQ1 = minus phase for auto-encoder
Q2,3: CA3 -> CA1 -> ECout  (ECin -> CA1 off) : ActM = minus phase for recall
Q4:   ECin -> CA1, ECin -> ECout (CA3 -> CA1 off, ECin -> CA1 on): ActP = plus phase for everything

[  q1      ][  q2  q3  ][     q4     ]
[ ------ minus ------- ][ -- plus -- ]
[   auto-  ][ recall-  ][ -- plus -- ]

	 DG -> CA3 -> CA1
	/    /      /    \

[----ECin---] -> [ ECout ]

minus phase: ECout unclamped, driven by CA1
auto-   CA3 -> CA1 = 0, ECin -> CA1 = 1
recall- CA3 -> CA1 = 1, ECin -> CA1 = 0

plus phase: ECin -> ECout auto clamped
CA3 -> CA1 = 0, ECin -> CA1 = 1
(same as auto- -- training signal for CA3 -> CA1 is what EC would produce!

ActQ1 = auto encoder minus phase state (in both CA1 and ECout

	used in EcCa1Path as minus phase relative to ActP plus phase in CHL)

ActM = recall minus phase (normal minus phase dynamics for CA3 recall learning)
ActP = plus (serves as plus phase for both auto and recall)

learning just happens at end of trial as usual, but encoder pathways use
the ActQ1, ActM, ActP variables to learn on the right signals

todo: implement a two-trial version of the code to produce a true theta rhythm
integrating over two adjacent alpha trials..
*/
package hip
