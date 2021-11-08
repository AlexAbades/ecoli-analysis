function [X, y] = selectTarget(X, att)
%SELECTTARGET We specify the attribute we want to predict.
%   We have to pass a table. and a string

% Select our target 
y = X(:,att); 
% Delete target from dataset 
X(:,att) = [];

% Convert tables into matrices 
y = table2array(y);
X = table2array(X);

% Binarize the attributes that have only two values 
X = TransformDataset(X);

end

