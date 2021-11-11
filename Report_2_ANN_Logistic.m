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

% K-fold cross validation 
K = 10;
CV = cvpartition(size(X,1),'Kfold', K);

N_test = size(X_test,1);
N_train = size(X_train,1);

% Parameters for neural network classifier
NHiddenUnits = 20;  % Number of hidden units



%% Fit multiclass neural network to training set
net = nc_main(X_train, y_train+1, X_test, y_test+1, NHiddenUnits);

%% Compute results on test data
% Get the predicted output for the test data
Y_test_est = nc_eval(net, X_test);

% Compute the class index by finding the class with highest probability from the neural
% network
y_test_est = max_idx(Y_test_est);
% Subtract one to have y_test_est between 0 and C-1
y_test_est = y_test_est-1;

% Compute error rate
ErrorRate = sum(y_test~=y_test_est)/N_test;
fprintf('Error rate: %.0f%%\n', ErrorRate*100);

%% Plot results
% Display trained network
mfig('Trained network'); clf;
displayNetworkClassification(net)

% Display decision boundaries
mfig('Decision Boundaries'); clf;
dbplot(X_test, y_test, @(X) max_idx(nc_eval(net, X))-1);
xlabel(attributeNames(1));
ylabel(attributeNames(2));
