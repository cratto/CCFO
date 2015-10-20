function [dsFeat,time] = extractFeaturesFromMNIST(dsRaw)

% [dsFeat,time] = extractFeaturesFromMNIST(dsRaw)
%
% This function extracts four types of features from the MNIST handwritten
% digit recognition data set: statistical features, principal components
% analysis, co-occurence features, and Sobel edge features. 
%
% The function requires installation of the Pattern Recognition Toolbox
% (PRT) for MATLAB:
%   http://covartech.github.io/ 
%
% INPUTS:
%   dsRaw: the PRT data set provided by prtDataGenMnist()
% OUTPUTS:
%   dsFeat: a new PRT data set containing the features that were computed
%   time:   the amount of time (in seconds) to compute each feature for
%           each observation in the data set.

% Author: Christopher R. Ratto, JHU/APL
% Date:   5 October 2015
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This software is Copyright 2015 The Johns Hopkins University Applied Physics Laboratory LLC
% All Rights Reserved
%
% This software is licensed to you under the terms of the Eclipse Public License, Version 1.0,
% a copy of which can be found at http://opensource.org/licenses/EPL-1.0.  Redistribution, 
% review, modification, and/or use of the software, in source and binary forms are ONLY permitted 
% provided you agree to and comply with the terms and conditions set forth in the license.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialize data structures and processors
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
PCA = prtPreProcPca('nComponents',10);  % Initialize the PCA preprocessor to use 10 components
PCA = PCA.train(dsRaw);                 % Train PCA on the entire data set
feats = [];                             % Features
time = [];                              % Extraction times

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Extract features from each sample
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i = 1:dsRaw.nObservations
    fprintf('Extracting features from sample %d of %d...\n',i,dsRaw.nObservations)
    
    % Statistical features
    tic
    statFeats = [mean(dsRaw.X(i,:)),std(dsRaw.X(i,:)),skewness(dsRaw.X(i,:)),kurtosis(dsRaw.X(i,:))];
    tStat = toc*ones(1,4);
    
    % PCA Features
    tic;
    dsPca = PCA.run(dsRaw.retainObservations(i));
    pcaFeats = dsPca.X;
    tPca = toc*ones(1,dsPca.nFeatures);

    % Co-occurence features
    tic;
    glcm1 = graycomatrix(reshape(dsRaw.X(i,:),28,28),'NumLevels',8);
    glcmProps1 = graycoprops(glcm1);
    glcm2 = graycomatrix(reshape(dsRaw.X(i,:),28,28),'NumLevels',16);
    glcmProps2 = graycoprops(glcm2);
    glcm3 = graycomatrix(reshape(dsRaw.X(i,:),28,28),'NumLevels',32);
    glcmProps3 = graycoprops(glcm3);
    glcmFeats = [glcmProps1.Contrast,glcmProps1.Correlation,glcmProps1.Energy,glcmProps1.Homogeneity,...
        glcmProps2.Contrast,glcmProps2.Correlation,glcmProps2.Energy,glcmProps2.Homogeneity,...
        glcmProps3.Contrast,glcmProps3.Correlation,glcmProps3.Energy,glcmProps3.Homogeneity];
    tGlcm = toc*ones(size(glcmFeats));
    
    % Sobel edge features
    tic;
    edgeDeg = [0, 45, 90, 135];
    edgeFeats = [];
    for j = 1:4
        H = fspecial('sobel');
        H = imrotate(H,edgeDeg(j));
        Y = imfilter(reshape(dsRaw.X(i,:),28,28),H);
        edgeFeats = [edgeFeats,trace(Y)];
        edgeFeats = [edgeFeats,trace(Y')];
        edgeFeats = [edgeFeats,sum(Y(14,:))];
        edgeFeats = [edgeFeats,sum(Y(:,14))];
    end
    tEdge = toc*ones(size(edgeFeats));

    % Concatenate the features into a single vector for the observation
    feats = [feats;statFeats,pcaFeats,glcmFeats,edgeFeats];
    time = [time;tStat,tPca,tGlcm,tEdge];

end
dsFeat = dsRaw;         % Copy the input PRT data set
dsFeat.X = feats;       % Overwrite the features

end

