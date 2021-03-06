% Create a .mat script to upload in python 
clear all; clc;

% Load data
load Ecoli_values.mat; 

% Clear variables except variable of interest.
clearvars -except X

% Drop the useless attributes (proteins names).
X(:,{'prot_name'}) = [];

% Delete the prdictible categories imS and imL because they have only two
% instances 
X_del = X;
classLabels_del = table2array(X_del(:,'cat'));
% Fiter the uniques values
[classNames_del, ~, ~] = unique(classLabels_del);
[~,y1_del] = ismember(classLabels_del, classNames_del);
y1_del = y1_del-1;
X_del(:,{'cat'}) = [];

X_del = [X_del array2table(y1,'VariableNames',{'cat'})];
idxs = X.cat == 2 ;
idxs1 = X.cat == 3;
idxs = idxs + idxs1; 

X_del(idxs == 1, :) = [];



% Put the binary attributes at the end and select targert
[X, y, names] = selectTarget(X, 'cat', 4);

% Transform the categories into numbers 
classLabels = y;
% Fiter the uniques values
[classNames, ia, ic] = unique(classLabels);
[~,y] = ismember(classLabels, classNames);

% Substract 1 to the vector so it's starts from 0.
y = y-1;

% Initialize some variables.
%Number of categories
C = max(y+1);
% Number of attribute 
M = size(X,2);
% Number of samples 
N = size(X,1);

attributeNames = names';

% Split randomly the data we can change it for a holdout and specify the 
% ratio of train and test set 
CV = cvpartition(N, 'Kfold', 2);

X_train = X(CV.training(1), :); 
X_test = X(CV.training(1), :);
y_train = y(CV.training(2), :);
y_test = y(CV.training(2), :);

% Get sizes of the test and training sets 
N_train = size(X_train, 1); 
N_test =  size(X_test, 1);

clearvars CV ia ic names classLabels

save('C:\Users\G531\Documents\1 - Universities\3 - DTU\002 - Course\02 - ML and Data mining\04 - IDE Extras\!origin\02450Toolbox_Python\02450Toolbox_Python\Data\Ecoli_python.mat');

% C:\Users\G531\Documents\1 - Universities\3 - DTU\002 - Course\02 - ML and Data mining\04 - IDE Extras\!origin\02450Toolbox_Python\02450Toolbox_Python\Data



