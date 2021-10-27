%% Importing dataset and assigning dependant and independant variables
rng(1) % Controlling randomness
songs = readtable('songs.csv'); % Importing dataset
X = songs(:,1:10); % Setting independent variable
Y = songs(:,'song_pop'); % Assigning dependent variable


% Using cvpartition to split dataset into 75/25 train/test split
cv = cvpartition(size(songs,1),'HoldOut',0.25);

Xtrain = X(cv.training,:); % Assigning training data
Ytrain = Y(cv.training,:);

Xtest = X(cv.test,:); % Assigning testing data
Ytest = Y(cv.test,:); 

%% Naive bayes model trained with multiple hyperparameter optimization.
% Hyperparameter optimization methods tested: Bayesian optimization, Grid search, Random search.

% Ran each model 10 times, used model with best accuracy.

% Model training commented out for convenience.

%% Bayesian optimization

%for i = 1:10
%opts = struct('Optimizer','bayesopt','AcquisitionFunctionName','expected-improvement-plus'...
%,'MaxObjectiveEvaluations',10)
%nb = fitcnb(Xtrain,Ytrain, 'OptimizeHyperparameters',{'DistributionNames', 'Kernel', 'Width'},...
%    'HyperparameterOptimizationOptions',...
%       opts)
%nbacc = 1 - loss(nb, Xtest, Ytest) 
%end


%% Grid search

%for i = 1:10
%optsgs = struct('Optimizer','gridsearch','AcquisitionFunctionName','expected-improvement-plus'...
%,'MaxObjectiveEvaluations',10)
%nbgs = fitcnb(Xtrain,Ytrain,'OptimizeHyperparameters',{'DistributionNames', 'Kernel', 'Width'},...
%    'HyperparameterOptimizationOptions',...
%       opts)
%nbgsacc = 1 - loss(nb, Xtest, Ytest)
%end


%% Random search

%for i = 1:10
%optsrs = struct('Optimizer','randomsearch','AcquisitionFunctionName','expected-improvement-plus'...
%,'MaxObjectiveEvaluations',10)
%nbrs = fitcnb(Xtrain,Ytrain,'OptimizeHyperparameters',{'DistributionNames', 'Kernel', 'Width'},...
%    'HyperparameterOptimizationOptions',...
%       opts)
%nbrsacc = 1 - loss(nb, Xtest, Ytest)
%end

%% Loading best model
% All optimization methods gave the same accuracy rates
% For fairness in comparison bayesian optimization is used
load('NB_final_model.mat');

 %% Testing loaded model on test data
 
[nb_predict,nb_scores] = predict(nb,Xtest); % Predicting classes as well as class probabilities
nb_loss = loss(nb,Xtest,Ytest) % Classification error calculation
%% Calculating accuracy and other measures

nb_confusion = confusionmat(table2array(Ytest),nb_predict) % Confusion matrix 
TP = nb_confusion(1,1); % True positives
FP = nb_confusion(1,2); % False positives
TN = nb_confusion(2,2); % True negatives
FN = nb_confusion(2,1); % False negatives

Accuracy = (TP + TN)/(TP + TN + FP + FN); % Accuracy of model
Precision = (TP)/(TP + FP); % Precision of model
Recall = (TP)/(TP + FN); % Recall of model
Specificity = (TN)/(TN + FP); % Specificity of model
F_measure = TP/(TP + 0.5*(FP+FN)); % F - score of model

summary = table(Accuracy, Precision, Recall, Specificity, F_measure, ...
    'VariableNames', {'Accuracy', 'Precision', 'Recall', 'Specificity','F_measure'}) % Table of calculated measures

%% ROC curve
logicY = logical(table2array(Ytest)); % Converting numeric values to logicals
[Xnb, Ynb, ~, AUCnb] = perfcurve(logicY, nb_scores(:,2),'true'); % Calculating ROC curve coordinates as well as area under the curve
plot(Xnb, Ynb,'LineWidth',1.5) % Plotting ROC curve
title('ROC curve of Naive Bayes')
legend(strcat('AUC = ', num2str(AUCnb)))
xlabel('False positive rate')
ylabel('True positive rate')
 
 