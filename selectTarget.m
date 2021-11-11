function [X, y, names] = selectTarget(X, att, c)
%SELECTTARGET We specify the attribute we want to predict.
%   We have to pass a table. and a string

% Select our target 
y = X(:,att); 
% Delete target from dataset 
X(:,att) = [];

if c == 1
    X_temp = X(:,{'lip', 'chg'});
    X(:,{'lip', 'chg'}) = [];
    X = [X X_temp];
else 
    X_temp = X(:,{'lip'});
    X(:,{'lip', 'chg'}) = [];
    X = [X X_temp];
end

names = X.Properties.VariableNames;
% Convert tables into matrices 
y = table2array(y);
X = table2array(X);

% Binarize the attributes that have only two values 
X = TransformDataset(X);

end

