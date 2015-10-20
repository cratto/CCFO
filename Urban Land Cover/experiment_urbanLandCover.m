function experiment_urbanLandCover(varargin)

% experiment_urbanLandCover.m
%
% This MATLAB script runs the feature selection experiment for the Urban
% Land Cover data set that was published in the following manuscript:
%   C.R. Ratto, C.A. Caceres, H.C. Schoeberlein, "Cost-Constrained Feature
%   Optimization in Kernel Machine Classifiers," IEEE Signal Processing
%   Letters, 2015.
%
% The script requires the Urban Land Cover data set, which is available
% from the UCI Machine Learning Repository:
%   https://archive.ics.uci.edu/ml/datasets/Urban+Land+Cover
%
% The script also requires installation of the Pattern Recognition Toolbox
% (PRT) for MATLAB:
%   http://covartech.github.io/ 
%
% INPUTS (optional):
%   dataDir:   full path to the directory in which the XLS data is saved
%              (default is the current working directory)
%
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
% Load in the training and test data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if nargin == 0
    dataDir = pwd;
else
    dataDir = varargin{1};
end
[~,~,raw] = xlsread(fullfile(dataDir,'training.csv'));             % Read raw data from XLS-converted CSV
X = cell2mat(raw(2:end,2:end));                                     % Features from spreadsheet
labels = strrep(raw(2:end,1),' ','');                               % Class labels from spreadsheet
uLabels = unique(labels);                                           % Unique class labels
Y = nan(size(labels));                                              % Numeric class labels 
for i = 1:length(labels)
    Y(i,:) = find(strcmp(labels{i},uLabels));                       % Assign a numeric label to each class name
end
featNames = raw(1,2:end);                                           % Feature names
dsTrain = prtDataSetClass(X,Y);                                     % Training PRT data set
dsTrain.classNames = uLabels;                                       % Training class names
dsTrain.featureNames = featNames;                                   % Training feature names
clear raw X labels uLabels Y featNames i

[~,~,raw] = xlsread(fullfile(dataDir,'testing.csv'));              % Read raw data from XLS-converted CSV
X = cell2mat(raw(2:end,2:end));                                     % Features from spreadsheet
labels = strrep(raw(2:end,1),' ','');                               % Class labels from spreadsheet
uLabels = unique(labels);                                           % Unique class labels
Y = nan(size(labels));                                              % Numeric class labels 
for i = 1:length(labels)
    Y(i,:) = find(strcmp(labels{i},uLabels));                       % Assign a numeric label to each class name
end
featNames = raw(1,2:end);                                           % Feature names
dsTest = prtDataSetClass(X,Y);                                      % Training PRT data set
dsTest.classNames = uLabels;                                        % Training class names
dsTest.featureNames = featNames;                                    % Training feature names
clear raw X labels uLabels Y featNames i

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Load in the feature extraction times
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[featCategories,~,categoryTimes] = featureTimes_urbanLandCover;       % Benchmarked feature computation times (see 'estimateFeatureTimes.m')
nFeatCategories = length(featCategories);                           % Number of feature categories
[categoryTimes,sortInds] = sort(cat(1,[],categoryTimes),'ascend');  % Sort the computation time of each feature category
featCategories = featCategories(sortInds);                          % Rearrange the order of feature category names
sortVec = [];                                                       % Vector of feature indices corresponding to the sorted categories
for iCategory = 1:nFeatCategories
    sortVec = [sortVec,find(~cellfun(@isempty,strfind(dsTest.featureNames,featCategories{iCategory})))]; %#ok<AGROW>
end
dsTrain.X = dsTrain.X(:,sortVec);                                   % Rearrange the order of features in the training set
dsTrain.featureNames = dsTrain.featureNames(:,sortVec);             % Rearrange the order of feature names in the training set
dsTest.X = dsTest.X(:,sortVec);                                     % Rearrange the order of features in the testing set
dsTest.featureNames = dsTest.featureNames(:,sortVec);               % Rearrange the order of feature names in the testing set
categoryInds = cell(1,nFeatCategories);                             % Feature indices corresponding to each category
T = nan(1,dsTrain.nFeatures);                                       % Computation time (seconds) of each feature
TperCategory = nan(1,nFeatCategories);                              % Total computation time (seconds) of each category
categoryIndBegin = nan(1,nFeatCategories);                          % The index of the first feature from each category
for iCategory = 1:nFeatCategories
    categoryInds{iCategory} = find(~cellfun(@isempty,strfind(dsTest.featureNames,featCategories{iCategory}))); % Find the features that belong to this category      
    categoryIndBegin(iCategory) = min(categoryInds{iCategory});
    T(categoryInds{iCategory}) = categoryTimes(iCategory);          % Computation time for each feature (seconds)
    TperCategory(iCategory) = sum(T(categoryInds{iCategory}));      % Computation time for each feature category (seconds)
end
tNorm = T./sum(T);                                                  % Normalized computation time of each feature
tNormCategory = TperCategory./sum(TperCategory);                    % Normalized computation time of each feature category
featCategories = strrep(featCategories,'_','');                     % Remove underscores from feature category names since it screws up the plots
clear categoryTimes iCategory i sortVec sortInds

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Set CCFO hyperparameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
a0 = linspace(0,2,500);
b0 = linspace(0,300,500);
tau = nan(500,500);
F = nan(500,500);
for i = 1:500
    for j = 1:500
        a = a0(i)*ones(size(tNorm));
        b = a0(i) + b0(j)*tNorm;
        tau(i,j) = sum(tNorm .* a./(a+b));
        F(i,j) = (1/dsTrain.nFeatures)*sum((a+1)./(a+b+1));
    end
end
desiredT = 0.25;                                                    % Expected runtime
desiredF = 0.50;                                                    % Maximum posterior probability of a feature being selected
dist = (tau-desiredT).^2 + (F-desiredF).^2;
[iMin,jMin] = find(dist == min(dist(:)));
a0 = a0(iMin);
b0 = b0(jMin);
a = a0*ones(size(tNorm));
b = a0 + b0*tNorm;
clear featInd legendNames colors tau F iA iB iFeat

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialize the classifiers
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% All classifiers will use the same kernel
kernel = prtKernelRbfNdimensionScale;                               % RBF kernel, scale the sigma parameter to dimensionality of the features
kernelSet = prtKernelDc & kernel;                                   % Add a bias dimension to the kernel (dc kernel)

% CCFO - Cost Constrained Feature Optimization
CCFO = prtClassCCFO('kernels',kernelSet,'pruneFeatures',false,'pruneObservations',false,'verbosePlot',false,'verboseText',true,'a',a','b',b','gamma',1,'ridge',10);    
algoCCFO = prtPreProcZmuv + prtClassBinaryToMaryOneVsAll('baseClassifier',CCFO); % Normalize features, One-vs-All classification since this is a multiclass problem

% RVM - Relevance Vector Machine
RVM = prtClassRvm('kernels',kernelSet);
algoRVM = prtPreProcZmuv + prtClassBinaryToMaryOneVsAll('baseClassifier',RVM);   % Normalize features, One-vs-All classification since this is a multiclass problem

% JCFO - Joint Classifier and Feature Optimization
JCFO = prtClassJCFO('kernels',kernelSet,'ridge',10,'pruneFeatures',false,'pruneObservations',false,'verboseText',1,'verbosePlot',0,'gamma1',1,'gamma2',1);
algoJCFO = prtPreProcZmuv + prtClassBinaryToMaryOneVsAll('baseclassifier',JCFO); % Normalize features, One-vs-All classification since this is a multiclass problem

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Train and test CCFO
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
trainedCCFO = algoCCFO.train(dsTrain);                                     % Train CCFO on the training set
dsOutCCFO = trainedCCFO.run(dsTest);                                       % Run CCFO on the test set
[~,dsOutCCFO.X] = max(dsOutCCFO.X,[],2);                                   % Change 'soft' decision values to 'hard' values (maximum a posteriori)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Train and test JCFO
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
trainedJCFO = algoJCFO.train(dsTrain);                                     % Train the one-vs-all JCFO
dsOutJCFO = trainedJCFO.run(dsTest);                                       % Run the one-vs-all JCFO
[~,dsOutJCFO.X] = max(dsOutJCFO.X,[],2);                                   % Change 'soft' decision values to 'hard' values (maximum a posteriori) 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Train and test RVM (individual feature categories)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
pcCategory = nan(1,nFeatCategories);                                       % Percent correct using each feature category
for iCategory = 1:nFeatCategories                                          % Loop over all feature categories
    dsCategoryTrain = dsTrain.retainFeatures(categoryInds{iCategory});     % Retain only features from this category for the training set
    dsCategoryTest = dsTest.retainFeatures(categoryInds{iCategory});       % Retain only features from this category for the testing set
    trainedRVM = algoRVM.train(dsCategoryTrain);                           % Train one-vs-all RVM
    dsOutCategory = trainedRVM.run(dsCategoryTest);                        % Run one-vs-all RVM
    [~,dsOutCategory.X] = max(dsOutCategory.X,[],2);                       % Change 'soft' decision values to 'hard' values (maximum a posteriori)  
    pcCategory(iCategory) = prtScorePercentCorrect(dsOutCategory);         % Calculate percent correct (accuracy overall) 
end
clear iCategory dsCategoryTrain dsCategoryTest dsOutCategory

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Train and test RVM (all features together)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
trainedRVM = algoRVM.train(dsTrain);                                % Train the one-vs-all RVM
dsOutRVM = trainedRVM.run(dsTest);                                  % Run the one-vs-all RVM
[~,dsOutRVM.X] = max(dsOutRVM.X,[],2);                              % Change 'soft' decision values to 'hard' values (maximum a posteriori)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot results
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot the prior on selecting features from each of the feature categories
% This will be a beta distribution over [0,1]. Features that take longer to
% compute should have higher probability of not being selected.
figure(1),set(gcf,'position',[610,512,700,441])
colors = prtPlotUtilClassColors(length(featCategories)); 
colors(7,:) = [0,1,1];
subplot(2,1,1),hold on
for iFeat = 1:length(featCategories(1:8))
    featInd = categoryInds{iFeat}(1);
    plot(linspace(0,1),betapdf(linspace(0,1),a(featInd),b(featInd)),'color',colors(iFeat,:),'linewidth',2);
    xlabel('\rho'),ylabel('p(\rho|a,b)')
end
axis tight
title('Cheap Features (Urban Land Cover)')
legend(featCategories(1:8),'location','southeastoutside')
subplot(2,1,2),hold on
for iFeat = 1:length(featCategories(9:15))
    featInd = categoryInds{8+iFeat}(1);
    plot(linspace(0,1),betapdf(linspace(0,1),a(featInd),b(featInd)),'color',colors(iFeat,:),'linewidth',2);
    xlabel('\rho'),ylabel('p(\rho|a,b)')
end
axis tight
title('Expensive Features (Urban Land Cover)')
legend(featCategories(9:15),'location','southeastoutside')
clear iFeat featInd colors

% Baseline - for each category, show feature computation and RVM performance
% Take-home point: most expensive features not always the best for
% classification performance
figure(2)     
[ha,h1,h2] = plotyy(1:nFeatCategories,100*pcCategory,1:nFeatCategories,tNormCategory);
h1.LineStyle = '-';h1.LineWidth = 2;h1.Marker = 'o';h1.MarkerSize = 8;
h2.LineStyle = '--';h1.LineWidth = 2;h1.Marker = '^';h1.MarkerSize = 8;
ha(1).XTick = 1:length(pcCategory); ha(1).XTickLabel = featCategories; ha(1).YLim = [0,100]; ha(1).YTick = 0:10:100; ha(1).XLim = [1,nFeatCategories]; ha(1).XTickLabelRotation = 30;
ha(2).XTick = 1:length(pcCategory); ha(2).XTickLabel = []; ha(2).YLim = [0,1]; ha(2).YTick = 0:0.1:1; ha(2).XLim = [1,nFeatCategories];
ylabel(ha(1),'Accuracy (% Correct)')
ylabel(ha(2),'Total Normalized Cost')
title('RVM Accuracy and Total Cost of Each Feature Category','FontSize',12)
clear ha h1 h2

% Plot the confusion matrices using all the features
% CCFO
figure(3),set(gcf,'outerposition',[65,301,1780,579])
h = subplot(1,3,1);
prtScoreConfusionMatrix(dsOutCCFO)
pcCCFO = prtScorePercentCorrect(dsOutCCFO);
h.XTickLabelRotation = 20;
title(['CCFO - ',num2str(100*pcCCFO,'%0.2f'),'% Correct'],'Fontsize',12)
axis square
% RVM
h = subplot(1,3,2);
prtScoreConfusionMatrix(dsOutRVM)                       
pcRVM = prtScorePercentCorrect(dsOutRVM);                        
title(['RVM - ',num2str(100*pcRVM,'%0.2f'),'% Correct'],'fontsize',12)
h.XTickLabelRotation = 20;
axis square
% JCFO 
h = subplot(1,3,3);
prtScoreConfusionMatrix(dsOutJCFO)   
pcJCFO = prtScorePercentCorrect(dsOutJCFO);
title(['JCFO - ',num2str(100*pcJCFO,'%0.2f'),'% Correct'],'fontsize',12)
h.XTickLabelRotation = 20;
axis square
clear h

% Compare feature selection performance
thetaCCFO = nan(dsTrain.nClasses,dsTrain.nFeatures);
thetaJCFO = nan(dsTrain.nClasses,dsTrain.nFeatures);
for iClass = 1:dsTrain.nClasses                                                                         % Loop over all one-vs-all classifiers
    thetaCCFO(iClass,:) = trainedCCFO.actionCell{2}.baseClassifier(iClass).theta';                      % CCFO feature selector parameters for this one-vs-all classifier
    thetaJCFO(iClass,:) = trainedJCFO.actionCell{2}.baseClassifier(iClass).theta';                      % JCFO feature selector parameters for this one-vs-all classifier
end
costReductionCCFO = nan(dsTrain.nClasses,1);
costReductionJCFO = nan(dsTrain.nClasses,1);
for iClass = 1:dsTrain.nClasses
    costReductionCCFO(iClass,:) = sum(tNorm(thetaCCFO(iClass,:)>=0.5));
    costReductionJCFO(iClass,:) = sum(tNorm(thetaJCFO(iClass,:)>=2*median(thetaJCFO(:))));
end
costReductionCCFO = mean(costReductionCCFO);
costReductionJCFO = mean(costReductionJCFO);

figure(4),set(gcf,'outerposition',[255,317,568,596])
h = subplot(2,1,1);
imagesc(thetaCCFO),colormap bone
h.YTick = 1:9; h.YTickLabel = dsTrain.classNames; h.XTick = categoryIndBegin; h.XTickLabel = featCategories; h.XTickLabelRotation = 70;
caxis([0,1]); h = colorbar; ylabel(h,'Feature Scaling Parameter (\theta)')
title({'CCFO: \theta Learned for Each One-vs-All Classifier',sprintf('Average Cost of Feature Extraction: %0.2f',costReductionCCFO)})
h = subplot(2,1,2);
imagesc(thetaJCFO),colormap bone
h.YTick = 1:9; h.YTickLabel = dsTrain.classNames; h.XTick = categoryIndBegin; h.XTickLabel = featCategories; h.XTickLabelRotation = 70;
caxis([0,2*median(thetaJCFO(:))]); h=colorbar; ylabel(h,'Feature Scaling Parameter (\theta)')
title({'JCFO: \theta Learned for Each One-vs-All Classifier',sprintf('Average Cost of Feature Extraction: %0.2f',costReductionJCFO)})
clear iClass h m

% Average number of features selected
avgNumFeatsSelectedRVM = dsTest.nFeatures;
avgNumFeatsSelectedJCFO = mean(sum(thetaJCFO > 2*median(thetaJCFO(:)),2));
avgNumFeatsSelectedCCFO = mean(sum(thetaCCFO > 0.5,2));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Print out summary of results
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('*************************************\n')
fprintf('Urban Land Cover Feature Set Summary\n')
fprintf('*************************************\n')
for iCategory = 1:nFeatCategories
    fprintf('%s \t %d \t %0.4f \t %0.2f\n',featCategories{iCategory},length(categoryInds{iCategory}),tNormCategory(iCategory),100*pcCategory(iCategory));
end
fprintf('*************************************\n')
fprintf('Urban Land Cover Performance Comparison\n')
fprintf('*************************************\n')
fprintf('Accuracy (RVM): %0.2f\n',100*pcRVM)
fprintf('Accuracy (JCFO): %0.2f\n',100*pcJCFO)
fprintf('Accuracy (CCFO): %0.2f\n',100*pcCCFO)
fprintf('Avg. # Features Selected (RVM): %0.2f\n',avgNumFeatsSelectedRVM)
fprintf('Avg. # Features Selected (JCFO): %0.2f\n',avgNumFeatsSelectedJCFO)
fprintf('Avg. # Features Selected (CCFO): %0.2f\n',avgNumFeatsSelectedCCFO)
fprintf('Avg. Relative Extraction Cost (RVM): 100\n')
fprintf('Avg. Relative Extraction Cost (JCFO) %0.2f\n',100*costReductionJCFO)
fprintf('Avg. Relative Extraction Cost (CCFO): %0.2f\n',100*costReductionCCFO)
keyboard
end


