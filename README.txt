Cost-Constrained Feature Optimization (CCFO) for MATLAB
 
This MATLAB code may be used to replicate the results published in the following manuscript:
   C.R. Ratto, C.A. Caceres, H.C. Schoeberlein, "Cost-Constrained Feature
   Optimization in Kernel Machine Classifiers," IEEE Signal Processing
   Letters, 2015.

The code is organized into three directories as follows:
	Urban Land Cover/ - Code for replicating the urban land cover experiment from the paper.
		experiment_urbanLandCover.m   - Script for running the experiment
		featureTimes_urbanLandCover.m - Function for estimating feature computation times
	MNIST/ - Code for replicating the MNIST experiment from the paper
		experiment_mnist.m         - Script for running the experiment
		extractFeaturesFromMNIST.m - Function for extracting features from the handwritten digits
	PRT Plugins/ - The actual machine learning code, written as a plugin to the Pattern Recognition Toolbox (PRT)
		prtClassJCFO - Joint Classifier and Feature Optimization
		prtClassCCFO - Cost Constrained Feature Optimization

To run either experiment, or to use our code in your own research, you must download and install the 
Pattern Recognition Toolbox (PRT) for MATLAB at http://covartech.github.io/

To run the urban land cover experiment, you must download the data from the UCI machine
learning repository: https://archive.ics.uci.edu/ml/datasets/Urban+Land+Cover

The code benchmarks each of the feature computation times on a test image of a black and white circle. 
The function "MidpointCircle.m" required to generate the test image for doing this is available via Matlab Central:
http://www.mathworks.com/matlabcentral/fileexchange/14331-draw-a-circle-in-a-matrix-image/content/MidpointCircle.m

*******************************************************************************************************************
This software is Copyright 2015 The Johns Hopkins University Applied Physics Laboratory LLC
All Rights Reserved

This software is licensed to you under the terms of the Eclipse Public License, Version 1.0,
a copy of which can be found at http://opensource.org/licenses/EPL-1.0.  Redistribution, 
review, modification, and/or use of the software, in source and binary forms are ONLY permitted 
provided you agree to and comply with the terms and conditions set forth in the license.
*******************************************************************************************************************