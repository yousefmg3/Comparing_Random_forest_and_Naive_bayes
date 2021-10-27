function createfigure1(ymatrix1, YMatrix2)
%CREATEFIGURE1(ymatrix1, YMatrix2)
%  YMATRIX1:  bar matrix data
%  YMATRIX2:  matrix of y data

%  Auto-generated by MATLAB on 11-Dec-2020 18:52:29

% Create figure
figure1 = figure;

% Create axes
axes1 = axes('Parent',figure1);
hold(axes1,'on');
colororder([0 0.447 0.741]);

% Activate the left side of the axes
yyaxis(axes1,'left');
% Create multiple lines using matrix input to bar
bar1 = bar(ymatrix1);
set(bar1(2),'DisplayName','Random Forest',...
    'FaceColor',[0.392156862745098 0.831372549019608 0.074509803921569]);
set(bar1(1),'DisplayName','Naive Bayes',...
    'FaceColor',[0.301960784313725 0.745098039215686 0.933333333333333]);

% Create ylabel
ylabel('Average Accuracy');

% Set the remaining axes properties
set(axes1,'YColor',[0 0.447 0.741]);
% Activate the right side of the axes
yyaxis(axes1,'right');
% Create multiple lines using matrix input to plot
plot1 = plot(YMatrix2,'LineWidth',1.5,'Color',[0 0 0]);
set(plot1(1),'DisplayName','Random Forest');
set(plot1(2),'DisplayName','Naive Bayes','LineStyle','--');

% Create ylabel
ylabel('Average time /s');

% Set the remaining axes properties
set(axes1,'YColor',[0.85 0.325 0.098]);
% Create title
title('Graph comparing average time and accuracy');

box(axes1,'on');
hold(axes1,'off');
% Set the remaining axes properties
set(axes1,'XTick',[1 2 3],'XTickLabel',...
    {'Bayesian optimization','Grid search','Random search'});
% Create legend
legend1 = legend(axes1,'show');
set(legend1,'Location','southwest','FontSize',6);
