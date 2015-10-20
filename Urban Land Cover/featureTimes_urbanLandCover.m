function [featureCategories,timeAbs,timeRel] = featureTimes_urbanLandCover()

% [timeAbs,timeRel] = featureTimes_urbanLandCover
%
% Function to estimate the extraction times for each of the features in the
% urban land cover data set which is available from the UCI Machine Learning Repository:
%   https://archive.ics.uci.edu/ml/datasets/Urban+Land+Cover
%
% The code benchmarks each of the feature computation times on a test image
% of a black and white circle. The function "MidpointCircle.m" required to
% generate the test image is available via Matlab Central:
%   http://www.mathworks.com/matlabcentral/fileexchange/14331-draw-a-circle-in-a-matrix-image/content/MidpointCircle.m
%
% The computations for each feature were derived from the following reference 
% (page references provided in the source code):
%   Definiens AG, "Definiens 5 Reference Book," Munich, Germany 2006.
%
% INPUTS: none
%
% OUTPUTS:
%   timeAbs: the absolute timeAbs to compute each feature (in sec) via tic/toc
%   timeRel: the relative timeAbs (%) to compute each feature in the set
%
% Author: Carlos A. Caceres, JHU/APL
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
% Initialize the test image
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
featureCategories = {'Area','Assym','BordLngth','BrdIndx','Bright','Compact','Dens','GLCM','LW','Mean','NDVI','Rect','Round','SD','ShpIndx'};
imgSize = 1024;                                                                 % Size of test image
img = zeros(imgSize);                                                           % Initialize the test image
img = MidpointCircle(img, 250, imgSize/2, imgSize/2, 1);                        % Fill in with a circle
img2 = img + randn(size(img));                                                  % Add white Gaussian noise

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Area (ref. page 58)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic;
area = sum(img(:)==1);  %assume each pixel has area = 1;
timeAbs(1) = toc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Assymmetry (ref. page 60)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic;
[idListX,idListY] = find(img==1);
varX = var(idListX);
varY = var(idListY);
varXY = var(idListX.*idListY);
assym = 2*sqrt(.25*(varX+varY)^2 + (varXY)^2 - varX*varY)/(varX+varY);
timeAbs(2) = toc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Border length (ref. page 36)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic;
temp = double(bwperim(img)==1);
bordlength = sum(temp(:));
timeAbs(3) = toc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Border index (ref. page 63)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic;
temp = double(bwperim(img)==1);
[idListX,idListY] = find(img==1);
P(1,:) = idListX;P(2,:) = idListY;
xbar = mean(idListX);
ybar = mean(idListY);
x = idListX - xbar;
y = -(idListY - ybar); % This is negative for the orientation calculation (measured in the counter-clockwise direction).
N = length(x);

% Calculate normalized second central moments for the region. 1/12 is
% the normalized second central moment of a pixel with unit length.
uxx = sum(x.^2)/N + 1/12;
uyy = sum(y.^2)/N + 1/12;
uxy = sum(x.*y)/N;

% Calculate major axis length, minor axis length, and eccentricity.
common = sqrt((uxx - uyy)^2 + 4*uxy^2);
l = 2*sqrt(2)*sqrt(uxx + uyy + common);
width = 2*sqrt(2)*sqrt(uxx + uyy - common);

borderIndex = sum(temp(:))/(2*(l+width));
timeAbs(4) = toc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Brightness (ref. page 42)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
img3 = repmat(img,1,1,3);
tic;
%     for an RGB image
wkb = [1,2,3];
brightness = (1/mean(wkb))*sum((wkb.*reshape(mean(mean(img3,1),2),1,3)));
%     % for a binary image
%     mean(img(:));
timeAbs(5) = toc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compactness (ref. page 63)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic;
[idListX,idListY] = find(img==1);
P(1,:) = idListX;P(2,:) = idListY;
xbar = mean(idListX);
ybar = mean(idListY);
x = idListX - xbar;
y = -(idListY - ybar); % This is negative for the orientation calculation (measured in the counter-clockwise direction).
N = length(x);

% Calculate normalized second central moments for the region. 1/12 is
% the normalized second central moment of a pixel with unit length.
uxx = sum(x.^2)/N + 1/12;
uyy = sum(y.^2)/N + 1/12;
uxy = sum(x.*y)/N;

% Calculate major axis length, minor axis length, and eccentricity.
common = sqrt((uxx - uyy)^2 + 4*uxy^2);
l = 2*sqrt(2)*sqrt(uxx + uyy + common);
width = 2*sqrt(2)*sqrt(uxx + uyy - common);

compactness = l*width/sum(img(:)==1);
timeAbs(6) = toc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Density (ref. page 62)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic;
[idListX,idListY] = find(img==1);
varX = var(idListX);
varY = var(idListY);
dens = sqrt(sum(img(:)==1))/(1+sqrt(varX+varY));
timeAbs(7) = toc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Gray-level co-ocurrence matrix
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic;
graycomatrix(img2);
timeAbs(8) = toc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Length/width (ref. page 59)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic;
[idListX,idListY] = find(img==1);
P(1,:) = idListX; P(2,:) = idListY;
xbar = mean(idListX);
ybar = mean(idListY);
x = idListX - xbar;
y = -(idListY - ybar); % This is negative for the orientation calculation (measured in the counter-clockwise direction).
N = length(x);

% Calculate normalized second central moments for the region. 1/12 is
% the normalized second central moment of a pixel with unit length.
uxx = sum(x.^2)/N + 1/12;
uyy = sum(y.^2)/N + 1/12;
uxy = sum(x.*y)/N;

% Calculate major axis length, minor axis length, and eccentricity.
common = sqrt((uxx - uyy)^2 + 4*uxy^2);
l = 2*sqrt(2)*sqrt(uxx + uyy + common);
width = 2*sqrt(2)*sqrt(uxx + uyy - common);

lw = l/width;
timeAbs(9) = toc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Mean (ref. page 40)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic;
datMean = mean(img2(:));
timeAbs(10) = toc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% NDVI
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic;
ndvi = sum((img + img2)./(img-img2));
timeAbs(11) = toc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Rectangular fit (ref. page 65)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%get outlines of each object
tic;
[idListX,idListY] = find(img==1);
xbar = mean(idListX);
ybar = mean(idListY);
x = idListX - xbar;
y = -(idListY - ybar); % This is negative for the orientation calculation (measured in the counter-clockwise direction).
N = length(x);

% Calculate normalized second central moments for the region. 1/12 is
% the normalized second central moment of a pixel with unit length.
uxx = sum(x.^2)/N + 1/12;
uyy = sum(y.^2)/N + 1/12;
uxy = sum(x.*y)/N;

% Calculate major axis length, minor axis length, and eccentricity.
common = sqrt((uxx - uyy)^2 + 4*uxy^2);
height = 2*sqrt(2)*sqrt(uxx + uyy + common);
width = 2*sqrt(2)*sqrt(uxx + uyy - common);
area = sum(img(:)==1);
SquareMetric = width/height;
if SquareMetric > 1,
    SquareMetric = height/width;  %make aspect ratio less than unity
end
SquareMetric = SquareMetric/area;
timeAbs(12) = toc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Roundness (ref. page 64)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic;
area = sum(img(:)==1);
temp = double(bwperim(img)==1);
perimeter = sum(temp(:));
roundness = 4*pi*area/perimeter^2;
timeAbs(13) = toc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Standard deviation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic;
stdDeviation = std(img(:));
timeAbs(14) = toc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Shape index (ref. page 62)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic;
temp = double(bwperim(img)==1);
shpindx = sum(temp(:))/(4*sqrt(sum(img(:)==1)));
timeAbs(15) = toc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Save results
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
timeRel = 100*timeAbs./repmat(sum(timeAbs),1,length(timeAbs));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Legend of feature names
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Class: Land cover class (nominal)
% BrdIndx: Border Index (shape variable)
% Area: Area in m2 (size variable)
% Round: Roundness (shape variable)
% Bright: Brightness (spectral variable)
% Compact: Compactness (shape variable)
% ShpIndx: Shape Index (shape variable)
% Mean_G: Green (spectral variable)
% Mean_R: Red (spectral variable)
% Mean_NIR: Near Infrared (spectral variable)
% SD_G: Standard deviation of Green (texture variable)
% SD_R: Standard deviation of Red (texture variable)
% SD_NIR: Standard deviation of Near Infrared (texture variable)
% LW: Length/Width (shape variable)
% GLCM1: Gray-Level Co-occurrence Matrix [i forget which type of GLCM metric this one is] (texture variable)
% Rect: Rectangularity (shape variable)
% GLCM2: Another Gray-Level Co-occurrence Matrix attribute (texture variable)
% Dens: Density (shape variable)
% Assym: Assymetry (shape variable)
% NDVI: Normalized Difference Vegetation Index (spectral variable)
% BordLngth: Border Length (shape variable)
% GLCM3: Another Gray-Level Co-occurrence Matrix attribute (texture variable) d

end