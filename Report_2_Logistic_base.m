clear all; clc;
% Load data
load Ecoli_values.mat; 

% Drop the useless attributes 
X(:,{'prot_name'}) = [];

% Get the attribtes names
attributeNames = [{'Offset'} X.Properties.VariableNames];

% Split our data into the attributes and target 
% Select our target 
y = X(:,'cat'); 
% Delete target from dataset 
X(:,'cat') = [];

% Convert tables into matrices 

X = table2array(X);

% Binarize the attributes that have only two values 
X = TransformDataset(X);

% Trnsform the categories into numbers 
classLabels = table2cell(y);
% Fiter the uniques values
[classNames, ia, ic] = unique(classLabels);

[~,y] = ismember(classLabels, classNames);
% Substract 1 to the vector so it's starts from 0.
y = y-1;

k = ceil(0.6*size(X,1));
CV = cvpartition(size(X,1),'Holdout', 0.3); % Specify a partition on 30%
idx = CV.test;

% Separate to training and test data
X_train = X(~idx,:);
y_train = y(~idx,:);
X_test = X(idx,:);
y_test = y(idx,:);

N_test = size(X_test,1);
N_train = size(X_train,1);


%% Fit multinomial regression model
Y_train=oneoutofk(y_train,C);
Y_test=oneoutofk(y_test,C);

W_est = mnrfit(X_train, Y_train);

%% Compute results on test data
% Get the predicted output for the test data
Y_test_est = mnrval(W_est, X_test);       

% Compute the class index by finding the class with highest probability from the multinomial regression model
[y_, y_test_est] = max(Y_test_est, [], 2);
% Subtract one to have y_test_est between 0 and C-1
y_test_est = y_test_est-1;

% Compute error rate
ErrorRate = sum(y_test~=y_test_est)/N_test;
fprintf('Error rate: %.0f%%\n', ErrorRate*100);

%% Plot results
% Display decision boundaries
mfig('Decision Boundaries'); clf;
dbplot(X_test, y_test, @(X) max_idx(mnrval(W_est,X))-1);
xlabel(attributeNames(1));
ylabel(attributeNames(2));
