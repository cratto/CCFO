classdef prtClassCCFO < prtClass
 % prtClassCCFO  Cost Constrained Feature Optimization
    %
    % This is a class written to be compatible the Pattern Recognition Toolbox
    % (PRT) for MATLAB. The PRT may be downloaded here:
    %        http://covartech.github.io/ 
    %
    %   CLASSIFIER = prtClassCCFO returns a CCFO classifier
    %
    %   CLASSIFIER = prtClassCCFO(PROPERTY1, VALUE1, ...) constructs a
    %   prtClass object CLASSIFIER with properties as specified by
    %   PROPERTY/VALUE pairs.
    %
    %   A prtClassCCFO object inherits all properties from the abstract class
    %   prtClass. In addition is has the following properties:
    %
    %   kernels                - A cell array of prtKernel objects specifying
    %                            the kernels to use (note CCFO only works
    %                            right now with RBF and polynomial kernels)
    %   verbosePlot            - Flag indicating whether or not to plot during
    %                            training
    %   verboseText            - Flag indicating whether or not to output
    %                            verbose updates during training
    %   learningMaxIterations  - The maximum number of iterations
    %   ridge                  - Regularization parameter for ridge regression
    %                            initialization of the weights
    %   gamma                  - Hyperparameter controlling the prior on
    %                            beta (regression weights)
    %   a                      - Hyperparameter controlling the prior on
    %                            theta (feature selectors)
    %   b                      - Hyperparameter controlling the prior on
    %                            theta (feature selectors)
    %   pruneFeatures          - Flag determining whether or not features
    %                            with a small enough theta should be
    %                            removed
    %   pruneObservations      - Flag determining whether or not
    %                            observations with a small enough beta should be removed
    %
    %   A prtClassCCFO also has the following read-only properties:
    %
    %   learningConverged  - Flag indicating if the training converged
    %   beta               - The regression weights, estimated during training  
    %   theta              - The feature scaling factors, estimated in training
    %   delta              - Term defined in (14) of CCFO paper
    %   omega              - Term defined in (13) of CCFO paper
    %   Q                  - The EM objective function being optimized 
    %   relevantFeats      - Indices of features determined to be relevant
    %   relevantObs        - Indices of observations determined to be relevant
    %
    %   A prtClassCCFO object inherits the TRAIN, RUN, CROSSVALIDATE and
    %   KFOLDS methods from prtAction. It also inherits the PLOT method
    %   from prtClass.
    %
    %   Reference:
    %       C.R. Ratto, C.A. Caceres, H.C. Schoeberlein, "Cost-Constrained
    %       Feature Optimization for Kernel Machine Classifiers," IEEE
    %       Signal Processing Letters, 2015.
    %
    %       B. Krishnapuram, A. Herermink, L. Carin, & M.A. Figueiredo, "A
    %       Bayesian approach to joint feature selection and classifier
    %       design," IEEE Trans. PAMI, vol. 26, no. 9, pp. 1105-1111, 2004.
    %
    %   Author: Christopher R. Ratto, JHU/APL
    %   Date:   7 October, 2015
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
% Define properties
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Private properties for internal PRT use
    properties (SetAccess=private)
        name = 'Cost Constrained Feature Optimization'                      % Full name of the classifier
        nameAbbreviation = 'CCFO';                                          % Abbreviated name
        isNativeMary = false;                                               % Cannot handle multi-class data
    end
    
    % Public properties for general use
    properties
        verbosePlot = false;                                                % Whether or not to plot during training
        verboseText = false;                                                % Whether or not to write text during training
        ridge = 1;                                                          % Ridge regression penalty (for initializing beta)
        gamma = 1;                                                          % Hyperparameter for beta
        a = 1;                                                              % Hyperparameter for theta
        b = 1;                                                              % Hyperparameter for theta
        kernels = prtKernelDc & prtKernelRbfNdimensionScale;                % Kernel function
        pruneFeatures = false;                                              % Flag for removing features as we go
        pruneObservations = false;                                          % Flag for removing observations as we go
    end
    
    % Hidden properties that should generally be left alone
    properties (Hidden = true)
        learningMaxIterations = 100;                                        % Maximum number of iteratoins
        learningConvergedThreshold = .0001;                                 % Threshold for whether learning has converged
        learningNormWeightsThresh = 0.001;                                  % Threshold for whether the weights aren't changing
        learningNormFeatSelectThresh = 0.001;                               % Threshold for whether feature selection has converged
        pruningThreshBeta = 0.0001;                                         % Threshold for removing observations
        pruningThreshTheta = 0.5;                                           % Threshold for removing features
        featuresRetained = [];                                              % List of features that were retained
        nMaxFminconEvals = 100;                                             % Number of steps for fmincon optimization
    end
    
    % Properties that may be accessed for monitoring the learning algorithm
    properties (SetAccess = 'protected',GetAccess = 'public')
        learningConverged = [];                                             % Whether or not the training converged
        beta = [];                                                          % The regression weights
        theta = [];                                                         % The feature scaling factors
        omega = [];                                                         % Equation 14 in Krishnapuram et al.
        Q = [];                                                             % The EM objective function
        relevantFeats = [];                                                 % List of relevant features
        relevantObs = [];                                                   % List of relevant observations
    end
    
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Error checking
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods
        % Allow for string, value pairs
        function Obj = prtClassCCFO(varargin)
            Obj = prtUtilAssignStringValuePairs(Obj,varargin{:});
        end
        
        % Make sure the kernel is compatible with JCFO
        function Obj = set.kernels(Obj,val)

            if ~(isa(val.kernelCell{2},'prtKernelRbf') || isa(val.kernelCell{2},'prtKernelRbfNdimensionScale') || isa(val.kernelCell{2},'prtKernelPolynomial')) && ~isa(val.kernelCell{1},'prtKernelDc')
                error('prt:prtClassJCFO:kernels','Kernel must be DC followed by RBF or Polynomial.');
            else
                Obj.kernels = val;
            end

        end
    end
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Training, testing, and helper functions (called by PRT train and run API)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods (Access = protected, Hidden = true)
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Training function (called by Obj.train)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function Obj = trainAction(Obj,DataSet)
            
            %%%%%%%%%% Get necessary classifier parameters %%%%%%%%%%%%
            X = DataSet.X;
            Y = DataSet.Y;
            N = size(X,1);
            P = size(X,2);
            beta = ones(N+1,1);
            theta = .9*ones(P,1);
            omega = ones(N+1,1);
            kernels = Obj.kernels;
            
            converged = false;
            iteration = 0;
            relevantFeats = true(P,1);
            relevantObs = true(N,1);
            relevantKernels = [true;relevantObs];
            while ~converged
                %%%%%%%%%%%% Iteration counter %%%%%%%%%%%%%%
                iteration = iteration + 1;
                if Obj.verboseText
                    fprintf('CCFO EM Iteration %d:\t',iteration)
                end
                Xrel = X(:,relevantFeats);
                Nrel = size(Xrel,1);
                Prel = size(Xrel,2);
                thetaRel = theta(relevantFeats);
                betaRel = beta(relevantKernels);
                aRel = Obj.a(relevantFeats);
                bRel = Obj.b(relevantFeats);
                
                %%%%%%%%%%%% M-step %%%%%%%%%%%%%%
                % Update the feature scaling factors
                if iteration > 1
                    if abs(thetaNormDiff) > Obj.learningNormWeightsThresh || isnan(thetaNormDiff)
                        opts = optimoptions(@fmincon,'Algorithm','interior-point','MaxFunEvals',Obj.nMaxFminconEvals,'GradObj','on','TypicalX',betarnd(ones(size(thetaRel)),ones(size(thetaRel))),'Display','iter-detailed','TolX',1e-4,'TolFun',1e-4);%'PlotFcns',{@optimplotx,@optimplotfval,@optimplotstepsize});
                        thetaRel = fmincon(@(x)Obj.calcQ(Xrel,kernels,v,omegaRel,x,relevantKernels,aRel,bRel),thetaRel,[],[],[],[],zeros(size(thetaRel)),ones(size(thetaRel)),[],opts);
                        theta(relevantFeats) = thetaRel;
                        thetaNormDiff = (norm(theta)-thetaNorm)./thetaNorm;
                    else
                        fprintf('Feature selection converged. Skipping constrained optimization.\n')
                    end
                else
                    thetaNormDiff = nan;
                end
                thetaNorm = norm(theta);
                
                % Apply scaling factors to features and re-compute the Gram
                % matrix via the kernel function
                XT = bsxfun(@times,Xrel,thetaRel');
                dsTmp = prtDataSetClass(XT,Y);
                kernels = train(kernels,dsTmp);
                H = kernels.run_OutputDoubleArray(dsTmp); % Gram matrix for the kernels-transformed features that have been selected so far
                H = H(:,relevantKernels);
           
                % Update the regression weights
                if iteration == 1
                    betaRel = inv(Obj.ridge*eye(size(H,2)) + H'*H)*H'*Y; % Initialize weights using ridge regression
                    beta(relevantKernels) = betaRel;
                    betaNormDiff = nan;
                else
                    betaRel = S*inv(eye(length(betaRel)) + S*H'*H*S)*S*H'*v;
                    beta(relevantKernels) = betaRel;
                    betaNormDiff = (norm(beta)-betaNorm)./betaNorm;
                end
                betaNorm = norm(beta);
                beta = beta./betaNorm;
                
               %%%%%%%%%%%% E-step %%%%%%%%%%%%%%
                v = nan(N,1);
                for i = 1:N
                    normFactor = (2*Y(i)-1)*normpdf(H(i,:)*betaRel,0,1)/normcdf((2*Y(i)-1)*H(i,:)*betaRel,0,1);
                    if isnan(normFactor)
                        normFactor = 0;
                    end
                    v(i,:) = H(i,:)*betaRel + normFactor; % Expected value of linear observation model
                end
                
                omegaRel = nan(length(betaRel),1);
                for i = 1:length(betaRel)
                    omegaRel(i,:) = Obj.gamma*abs(betaRel(i))^(-1); % Expected value of weight variance
                end
                omega(relevantKernels) = omegaRel;
                S = diag(omegaRel.^(-1/2));
                
                % Recompute the expected log-posterior
                Q(iteration) = Obj.calcQ(Xrel,kernels,v,omegaRel,thetaRel,relevantKernels,aRel,bRel);
                
                %%%%%%%%%%%% Prune deselected training points and/or features, if enabled %%%%%%%%%%%%%%
                if Obj.pruneFeatures
                    relevantFeats(theta < Obj.pruningThreshTheta) = false;
                    theta(~relevantFeats) = 0;
                    thetaRel = theta(relevantFeats);
                end
                
                if Obj.pruneObservations
                    relevantObs(abs(beta) < Obj.pruningThreshBeta) = false;
                    relevantKernels = [true;relevantObs];
                    beta(~relevantKernels) = 0;
                    betaRel = beta(relevantKernels);
                    omegaRel = omega(relevantKernels);
                    S = diag(omegaRel.^(-1/2));
                end
                
                % For debugging purposes, plot how all of the parameters are updating
                if Obj.verbosePlot
                    figure(666)
                    subplot(2,3,1),plot(v),title('v'),axis tight
                    subplot(2,3,2),plot(log(omega),'marker','o'),title('log(\omega)'),axis tight
                    subplot(2,3,4),plot(beta,'marker','o'),title('\beta'),axis tight
                    subplot(2,3,5),plot(theta,'marker','o'),title('\theta'),axis tight
                    subplot(2,3,6),plot(Q),title('-E[log p(\beta,\theta|-)]'),axis tight
                    drawnow
                end
                
                %%%%%%%%%%%% Check for convergence %%%%%%%%%%%%%%
                if iteration == 1
                    Qdiff = nan;
                else
                    Qdiff = (Q(iteration)-Q(iteration-1))./Q(iteration-1);
                end
                if Obj.verboseText
                    fprintf('Q = %0.4f (diff = %0.4f)\t ||beta|| = %0.4f (diff = %0.4f)\n',Q(iteration),Qdiff,betaNorm,betaNormDiff)
                end
                
                if abs(Qdiff) < Obj.learningConvergedThreshold;
                    converged = true;
                    fprintf('Expected log-posterior converged within threshold, exiting.\n')
                elseif iteration == Obj.learningMaxIterations;
                    converged = true;
                    fprintf('Maximum number of iterations reached, exiting.\n')
                elseif abs(betaNormDiff) < Obj.learningNormWeightsThresh
                    converged = true;
                    fprintf('Magnitude of weight vector converged within threshold, exiting.\n')
                end
      
            end
            %%%%%%%%%% Save out learned parameters %%%%%%%%%%%%
            Obj.beta = beta;
            Obj.theta = theta;
            Obj.omega = omega;
            Obj.Q = Q(end);
            
            XT = bsxfun(@times,X,theta');
            dsTmp = prtDataSetClass(XT,Y);
            Obj.kernels = train(kernels,dsTmp);
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Running function (called by Obj.run)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function DataSetOut = runAction(Obj,DataSet)
            %%%%%%%%%% Get necessary classifier parameters %%%%%%%%%%%%
            X = DataSet.X;
            kernels = Obj.kernels;
            theta = Obj.theta;
            beta = Obj.beta;
            
            %%%%%%%%%% Run CCFO on dataset %%%%%%%%%%%%
            DataSet.X = bsxfun(@times,X,theta');
            H = kernels.run_OutputDoubleArray(DataSet);
            
            %%%%%%%%%% Build output dataset %%%%%%%%%%%%
            DataSetOut = DataSet;
            DataSetOut.X = normcdf(H*beta);
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Function for calculating the EM objective, Q and its derivative (called by Obj.trainAction)
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function [Qout,dQdTout,beta,omega] = calcQ(Obj,X,kernel,v,omega,theta,relevantKernels,a,b)

            % Build gram matrix using the proposed vector of feature scaling factors
            N = size(X,1);
            Nk = sum(relevantKernels);
            P = size(X,2);
            XT = bsxfun(@times,X,theta');
            dsTmp = prtDataSetClass(XT);
            kernels = train(kernel,prtDataSetClass(XT));
            H = kernels.run_OutputDoubleArray(dsTmp); % Gram matrix for the kernels-transformed features that have been selected so far
            H = H(:,relevantKernels);
            S = diag(omega.^(-1/2));
            
            % Calcualte the expected log-posterior
            beta = S*inv(eye(Nk) + S*(H'*H)*S)*S*H'*v;
            
            % Calcualte the expected log-posterior
            nu1 = psi(a) - psi(a + b);
            nu2 = psi(b) - psi(a + b);
            Q = -beta'*(H'*H)*beta + 2*beta'*H'*v - beta'*diag(omega)*beta + sum(theta.*nu1 + (1-theta).*nu2);
            Qout = -Q;
            
            % Calculate derivative of Q w.r.t. each theta
            dQdT = nan(1,P);
            if isa(Obj.kernels.kernelCell{2},'prtKernelPolynomial')
                n = Obj.kernels.kernelCell{2}.d;
                xTx = X*diag(theta)*X';
                for k = 1:P
                    xxk = X(:,k)*X(:,k)';
                    dHdT = [zeros(N,1),(n*(1+xTx).^(n-1)).*xxk]; % Derivative of polynomial kernel provided in Kirshnapuram et al., RECOMB '03
                    dQdT(k) = nu2(k) - nu1(k) - 2*sum(sum(((H*beta-v)*beta').*dHdT));
                end
                dQdTout = -dQdT;
            elseif isa(Obj.kernels.kernelCell{2},'prtKernelRbf')
                for k = 1:P
                    Xk = X(:,k);
                    dXk = repmat(sum((Xk.^2), 2), [1 N]) + repmat(sum((Xk.^2),2), [1 N]).' - 2*Xk*(Xk.');
                    if isa(Obj.kernels.kernelCell{2},'prtKernelRbfNdimensionScale')
                        dXk = dXk./(P*Obj.kernels.kernelCell{2}.sigma.^2);
                    else
                        dXk = dXk./Obj.kernels.kernelCell{2}.sigma.^2;
                    end
                    if isa(Obj.kernels.kernelCell{1},'prtKernelDc')
                        dXk = [zeros(N,1),dXk];
                    end
                    dHdT = -H.*dXk;
                    dQdT(k) = nu1(k) - nu2(k) - 2*sum(sum(((H*beta-v)*beta').*dHdT));
                    dQdTout(k) = -dQdT(k);
                end
            end
        end
    end
end
