%% Importing dataset and assigning dependant and independant variables
rng(1)
songs = readtable('songs.csv');
X = songs(:,1:10);
Y = songs(:,'song_pop');

% Partitioning to 75/25 split
cv = cvpartition(size(songs,1),'HoldOut',0.25)
Xtrain = X(cv.training,:);
Xtest = X(cv.test,:);
Ytrain = Y(cv.training,:);
Ytest = Y(cv.test,:); 

%% Initial Analysis
% Compare mean and std of variables based on whether they are popular or
% not
pop = songs(songs.song_pop == 1,:);
npop = songs(songs.song_pop == 0,:);

%% Mean and std of each audio feature based on if song is popular or not
for i = 1:10
    variable(i) = (pop.Properties.VariableNames(i));
    meanpop(i) = (mean(pop{:,i}));
    stdpop(i) = (std(pop{:,i}));
    meanNpop(i) = (mean(npop{:,i}));
    stdNpop(i) = (std(npop{:,i})); 
end

statistic = table(variable(:),meanpop(:),meanNpop(:),stdpop(:),stdNpop(:),...
    'VariableNames',{'Audio Feature','Mean pop','Mean not pop', 'Std pop','Std not pop'})
%% Histogram comparing artist_pop in popular vs non popular songs

histogram(pop{:,10})
hold on 
histogram(npop{:,10},'facecolor','#2ca25f')
legend('Popular','Not Popular')
title('artist\_pop')
xlabel('Artist Popularity')
xlim([0 1])
ylabel('Frequency')
hold off

%% Histogram comparing year in popular vs non popular songs

histogram(pop{:,8})
hold on 
histogram(npop{:,8},'facecolor','#2ca25f')
legend('Popular','Not Popular','Location','northwest')
xlabel('Year')
ylabel('Frequency')
title('year')
xlim([1950 2013])
hold off

%% Histogram comparing duration in popular vs non popular songs
histogram(pop{:,1})
hold on 
histogram(npop{:,1},'facecolor','#2ca25f')
legend('Popular','Not Popular')
title('duration')
xlabel('Time in seconds')
ylabel('Frequency')
xlim([0 1000])
hold off

%% Loading models

load('RF_final_model.mat');
load('NB_final_model.mat');

%%  Comparison of accuracy and other measures

rf_pred = predict(rf,Xtest);
nb_pred = predict(nb,Xtest);
predictions = [str2num(cell2mat(rf_pred)), nb_pred];
for i = 1:2
    confusion = confusionmat(table2array(Ytest),predictions(:,i)); % Confusion matrix 
    TP = confusion(1,1); % True positives
    FP = confusion(1,2); % False positives
    TN = confusion(2,2); % True negatives
    FN = confusion(2,1); % False negatives
    
    Accuracy(i) = (TP + TN)/(TP + TN + FP + FN); % Accuracy of model
    Precision(i) = (TP)/(TP + FP); % Precision of model
    Recall(i) = (TP)/(TP + FN); % Sensitivity of model
    Specificity(i) = (TN)/(TN + FP); % Specificity of model
    F_measure(i) = TP/(TP + 0.5*(FP+FN)); % F - score of model
end
summary = table({'Random Forest'; 'Naive Bayes'},Accuracy(:), Precision(:), Recall(:), Specificity(:), F_measure(:), ...
        'VariableNames', {'Model','Accuracy', 'Precision', 'Recall', 'Specificity', 'F_measure'}) % Table of calculated measures

%% ROC curves
[~,scoresrf] = predict(rf,Xtest); % Predicting classes as well as class probabilities 
[~,scoresnb] = predict(nb,Xtest); % Predicting classes as well as class probabilities 
logicY = logical(table2array(Ytest)); % Converting numeric values to logicals
[Xrf, Yrf, ~, AUCrf] = perfcurve(logicY, scoresrf(:,2),'true'); % Calculating ROC curve coordinates as well as area under the curve
[Xnb, Ynb, ~, AUCnb] = perfcurve(logicY, scoresnb(:,2),'true'); % Calculating ROC curve coordinates as well as area under the curve
plot(Xrf, Yrf,'LineWidth',2) 
hold on
plot(Xnb,Ynb,'color','#2ca25f', 'LineWidth',2)
legend(strcat('Random Forest = ', num2str(AUCrf)),strcat('Naive Bayes = ', num2str(AUCnb)),'Location','SE')
title('ROC curve of RF and NB')
xlabel('False positive rate')
ylabel('True positive rate')

