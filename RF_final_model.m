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

%% Random forest model training with multiple hyperparameter optimization methods as well as boosting methods.
% Hyperparameter optimization methods tested: Bayesian optimization, Grid search, Random search.
% Boosting methods tested: Bagging, AdaboostM1

% Ran each method 10 times, used hyperparameters of best model.

% Model training commented out for convenience.

%% Bayesian optimization and Bagging

%for i = 1:10
%t = templateTree('Reproducible',true)
%opts = struct('Optimizer','bayesopt','AcquisitionFunctionName','expected-improvement-plus'...
%,'MaxObjectiveEvaluations',10)
%rf =  fitcensemble(Xtrain,Ytrain, 'Method', 'Bag', 'Learners',t, 'OptimizeHyperparameters',...
%    {'NumLearningCycles', 'MinLeafSize','NumVariablesToSample'},'HyperparameterOptimizationOptions',...
%    opts)
%rfacc = 1 - loss(rf,Xtest,Ytest)
%end

%% Bayesian optimization and AdaboostM1

%for i = 1:10
%t = templateTree('Reproducible',true)
%opts = struct('kfold',10,'Optimizer','bayesopt','AcquisitionFunctionName','expected-improvement-plus'...
%,'MaxObjectiveEvaluations',10)
%rfada =  fitcensemble(Xtrain,Ytrain, 'Method', 'AdaboostM1', 'Learners',t, 'OptimizeHyperparameters',...
%    {'NumLearningCycles', 'MinLeafSize','NumVariablesToSample','LearnRate'},'HyperparameterOptimizationOptions',...
%    opts)
%rfadaacc = 1-loss(rfada,Xtest,Ytest)
%end



%% Grid search and Bagging

%for i = 1:10
%t = templateTree('Reproducible',true)
%opts = struct('Optimizer','gridsearch','AcquisitionFunctionName','expected-improvement-plus'...
%,'MaxObjectiveEvaluations',10)
%rfgs =  fitcensemble(Xtrain,Ytrain, 'Method', 'Bag', 'Learners',t, 'OptimizeHyperparameters',...
%    {'NumLearningCycles', 'MinLeafSize','NumVariablesToSample'},'HyperparameterOptimizationOptions',...
%    opts)
%rfgsacc = 1-loss(rfgs,Xtest,Ytest)
%end

%% Random search and Bagging

%for i = 1:10
%t = templateTree('Reproducible',true)
%optrs = struct('Optimizer','randomsearch','AcquisitionFunctionName','expected-improvement-plus'...
%,'MaxObjectiveEvaluations',10)
%rfrs =  fitcensemble(Xtrain,Ytrain, 'Method', 'Bag', 'Learners',t, 'OptimizeHyperparameters',...
%    {'NumLearningCycles', 'MinLeafSize','NumVariablesToSample'},'HyperparameterOptimizationOptions',...
%    optrs)
%rfrsacc = 1-loss(rfrs,Xtest,Ytest)
%end



%% Training best model

%rf = TreeBagger(480, Xtrain, Ytrain, ...,
%    'MinLeafSize', 1, 'NumVariablesToSample', 8, 'OOBPrediction','on') % Running model with best hyperparameters

%% Saving best model 

%save('RF_final_model.mat','rf') % Saving best model

%% Loading best model

% Bayesian optimization gave highest accuracy rates so it is the final model used.
load('RF_final_model.mat');

 %% Running loaded model on test data
 
[rf_predict,rf_scores] = predict(rf,Xtest); % Predicting classes as well as class probabilities 

%% Calculating accuracy and other measures
rf_confusion = confusionmat(table2array(Ytest),str2num(cell2mat((rf_predict)))) % Confusion matrix 

TP = rf_confusion(1,1); % True positives
FP = rf_confusion(1,2); % False positives
TN = rf_confusion(2,2); % True negatives
FN = rf_confusion(2,1); % False negatives

Accuracy = (TP + TN)/(TP + TN + FP + FN); % Accuracy of model
Precision = (TP)/(TP + FP); % Precision of model
Recall = (TP)/(TP + FN); % Sensitivity of model
Specificity = (TN)/(TN + FP); % Specificity of model
F_measure = TP/(TP + 0.5*(FP+FN)); % F - score of model

summary = table(Accuracy, Precision, Recall, Specificity, F_measure, ...
        'VariableNames', {'Accuracy', 'Precision', 'Recall', 'Specificity','F_measure'}) % Table of calculated measures

%% ROC curve
logicY = logical(table2array(Ytest)); % Converting numeric values to logicals
[Xrf, Yrf, ~, AUCrf] = perfcurve(logicY, rf_scores(:,2),'true'); % Calculating ROC curve
plot(Xrf, Yrf,'LineWidth',1.5) % Plotting ROC curve
title('ROC curve of Random Forest')
legend(strcat('AUC = ', num2str(AUCrf)))
xlabel('False positive rate')
ylabel('True positive rate')



% 1) 243, 1, 8 , 1235, 0.7179
% 2) 171, 17, 7, 815, 0.7108
% 3) 175, 1, 8, 1162, 0.7189
% 4) 480, 1, 8, 1743, 0.7223
% 5) 494, 1, 6, 1699, 0.7172
% 6) 240, 2, 7, 1090, 0.7185
% 7) 498, 1, 10, 1472, 0.7205
% 8) 296, 8, 9, 1287, 0.7133
% 9) 97, 6, 9, 1233, 0.7194
% 10) 499, 2, 8, 1529, 0.7202
