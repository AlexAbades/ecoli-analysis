clear all;clc;
%% Load data 
load Ecoli_values.mat; 

% Drop the useless attributes (proteins names) 
X(:,{'prot_name'}) = [];
classLabels = table2array(X(:,'cat'));
% Fiter the uniques values
[classNames, ia, ic] = unique(classLabels);
[~,y1] = ismember(classLabels, classNames);
y1 = y1-1;
X(:,{'cat'}) = [];

X = [X array2table(y1,'VariableNames',{'cat'})];
idxs = X.cat == 2 ;
idxs1 = X.cat == 3;
idxs = idxs + idxs1; 

X(idxs == 1, :) = [];



% Put the binary attributes at the end and select targert
[X, y, names] = selectTarget(X, 'cat', 2);

% % Transform the categories into numbers 
% classLabels = y;
% % Fiter the uniques values
% [classNames, ia, ic] = unique(classLabels);
% [~,y] = ismember(classLabels, classNames);
% % Select the number of classes e have
% 
% % Substract 1 to the vector so it's starts from 0.
% y = y-1;

C = max(y+1);
attributeNames = [{'Offset'} names];




%% 2-fold Cross Validation 
% BASIC PARAMETERS 
% Specify folders outter loop 
K = 10;
% SPecify folders inner loop 
KK = 10;
% Add ones to the matrix for the w0 or intercept values (Offset)
X=[ones(size(X,1),1) X];
% Select the number of models we are gonna test 
M= size(X,2);
% Select an initial guess for lambda values 
lambda_tmp=10.^(-2:1);
% Get the number of labdas we are going to test 
T=length(lambda_tmp);
% Initialize the number of neurons it's going to have ANN
h_tmp = [4 6 10 25 50 100];

% DATA TRANFORMATION 
% Split the data into K folds 
N = size(X,1);
CV = cvpartition(N, 'Kfold', K);


%%
Training = array2table(CV.training(1), 'VariableNames',{'Xtraining_1'});
for i=2:K
    a = strcat( 'Xtraining_',num2str(i) );
    Training = [Training array2table(CV.training(i), 'VariableNames',{a})];
end

Test = array2table(CV.test(1), 'VariableNames',{'Xtraining_1'});
for i=2:K
    a = strcat( 'Xtest_',num2str(i) );
    Test = [Test array2table(CV.test(i), 'VariableNames',{a})];
end

Trainingmat = table2array(Training);
Testmat = table2array(Test);
save('C:\Users\G531\Documents\1 - Universities\3 - DTU\002 - Course\02 - ML and Data mining\04 - IDE Extras\!origin\02450Toolbox_Python\02450Toolbox_Python\Data\Ecoli_TrainTestsplit.mat',...
    'CV','Training','Test', 'Trainingmat', 'Testmat');
save('C:\Users\G531\Documents\8 - Github\ecoli-analysis\Ecoli_TrainTestsplit.mat',...
    'CV','Training','Test', 'Trainingmat', 'Testmat');


%%
% Initialize Error rates: MULTINOMIAL REGRESSION 
% Outer Loop
ErrorRate_train = nan(K,1);
ErrorRate_test = nan(K,1);
% Inner Loop
ErrorRate_train2 = nan(T,KK);
ErrorRate_test2 = nan(T,KK);
% without features (baseline?)
ErrorRate_train_nofeatures = nan(K,1);
ErrorRate_test_nofeatures = nan(K,1);

wSoftmax = nan(K,T);

% Initialize Error rates: ARTIFICIAL NAURAL NETWORK
% Inner Loop 
ErrorRate_test_ann2 = nan(length(h_tmp),KK);
% Outer Loop
ErrorRate_test_ann = nan(K,1);

% OPTIMAL LAMBDA & NEURONS
lambda_opt = nan(K,1);
h_opt = nan(K,1);

% Initialize mean ans standard deviation for each train set
mu = nan(K, M-1); % Minus 3 to don't get the offset and the binary
sigma = nan(K, M-1);

% SHUFFLE IID SAMPLING 

% 2-Cross Validation Algorithm
% For each crossvalidation fold
for k = 1:K
    fprintf('Crossvalidation fold %d/%d\n', k, K);
    
    % Extract the training and test set
    X_train = X(CV.training(k), :);
    y_train = y(CV.training(k));
    X_test = X(CV.test(k), :);
    y_test = y(CV.test(k));
    
    % Use 10-fold crossvalidation to estimate optimal value of lambda
    % KK = 10;
    % Split Data in the into kk folders for the inner loop
    CV2 = cvpartition(size(X_train,1), 'Kfold', KK);
    for kk=1:KK
        tic
        fprintf('Inner fold %d/%d\n', kk, KK);
        X_train2 = X_train(CV2.training(kk), :);
        y_train2 = y_train(CV2.training(kk));
        X_test2 = X_train(CV2.test(kk), :);
        y_test2 = y_train(CV2.test(kk));
        
        % Standardize the training and test set based on training set in
        % the inner fold. Not standarize the binary attributes, set sigma
        % value really low so it doesn't compute NaN values
        % Our binary attributes are at the last two columns, 6 and 7
        %{
            mu2 = mean(X_train2(:,2:end));
            sigma2 = std(X_train2(:,2:end));
            X_train2(:,2:end) = (X_train2(:,2:end) - mu2) ./ sigma2;
            X_test2(:,2:end) = (X_test2(:,2:end) - mu2) ./ sigma2;
        %}
        mu2 = mean(X_train2(:,2:5));
        sigma2 = std(X_train2(:,2:5));
        X_train2(:,2:5) = (X_train2(:,2:5) - mu2) ./ sigma2+10^-8;
        X_test2(:,2:5) = (X_test2(:,2:5) - mu2) ./ sigma2+10^-8;

        
        % MULTINOMIAL REGULARIZED REGRESSION
        
        % Fit the multinomial regression for each lambda
        for t = 1:T
            
            % Define the softmax loss for multinomial logistic regression
            funObj = @(W)SoftmaxLoss2(W, X_train2, y_train2+1,C-1);
            % Setup the reguliarzation parameter
            regularization_strength = lambda_tmp(t);
            % Create a matrix for the lambdas
            lambda2 = regularization_strength*ones(M+1,C-1);
            lambda2(t,:) = 0; % Don't penalize biases
            options = [];
            kk= 1
            % Train the model using fminunc
            fprintf('Training multinomial logistic regression model %d...\n', t);
            wSoftmax(kk,t) = fminunc(@penalizedL2,zeros((M+1)*(C-1),1), ...
                options,funObj,lambda2(:)); % Changing in size. Should we initialize it beforehand? 
            % Reshape the predicted weigths 
            wSoftmax(kk,t) = reshape(wSoftmax(kk,t),[M+1 C-1]);
            wSoftmax(kk,t) = [wSoftmax(kk,t) zeros(M+1,1)]; 
            
            % Predict the class of the test set using the fitted model
            % Train
            [~, y_train_est2] = max([ones(length(X_train2),1) X_train2]*wSoftmax(kk,t),[],2);
            ErrorRate_train2(t,kk) = sum(y_train2 ~= y_train_est2)/length(y_train2);
            
            % Test 
            [~, y_test_est2] = max([ones(length(X_test2),1) X_test2]*wSoftmax(kk,t),[],2);
            ErrorRate_test2(t,kk) = sum(y_test2 ~= y_test_est2)/length(y_test2);
            
        end
      
        
        % ARTIFICIAL NEURAL NETWORKS
        for t = 1:length(h_tmp)
            fprintf('ANN: %d/%d\n', t, length(h_tmp));
            % We don't have to pass the offset values to the ANN 
            net = nc_main(X_train2(:,2:end), y_train2+1, X_test2(:,2:end), y_test2+1, h_tmp(t)); % Matrix must be positive semidefinite
            y_test_est2 = net.t_est_test;
            ErrorRate_test_ann2(t,kk) = sum(y_test2 ~= y_test_est2)/length(y_test2);
            
        end
        
        % MULTINOMIAL REGULARIZED REGRESSION 
        % Select optimal values for lambda
        [~,lambda_idx]  = min(sum(ErrorRate_test2,2)/KK);
        lambda_opt(k)   = lambda_tmp(lambda_idx);
        
        
        % ARTIFICIAL NEURAL NETWORKS
        % Select optimal values of Neurons, h.
        [~,h_idx] = min(sum(ErrorRate_test2,2)/KK);
        h_opt(k) = h_tmp(h_idx);
        
        % FEATURE TRANFORMATION 
        % Standardize datasets in outer fold.
        
        %STANDARD
        mu(k,:) = mean(X_train(:,2:end));
        sigma(k,:) = std(X_train(:,2:end));
        X_train_std = X_train;
        X_test_std = X_test;
        X_train_std(:,2:end) = (X_train(:,2:end) - mu(k,:)) ./ sigma(k,:);
        X_test_std(:,2:end) = (X_test(:,2:end) - mu(k,:)) ./ sigma(k,:);
        
        % Standarization No binary
        %{
        mu(k,  :) = mean(X_train(:,2:5));
        sigma(k, :) = std(X_train(:,2:5));
        
        X_train_std = X_train;
        X_test_std = X_test;
        X_train_std(:,2:5) = (X_train(:,2:5) - mu(k , :))./sigma(k, :)+10^-8;
        X_test_std(:,2:5) = (X_test(:,2:5) - mu(k, :))./sigma(k, :)+10^-8;
        %}
            
            
        % MULTIVARIATE REGULARIZED REGRESSION
        
%         % Check the test error of the model with the optimal value of
%         % lambda selected on the inner fold.
%         
%         % Define the softmax loss for multinomial logistic regression
%         funObj = @(W)SoftmaxLoss2(W, X_train, y_train+1,C);
%         % Setup the reguliarzation parameter
%         regularization_strength = lambda_opt(k);
%         % Create a matrix for the lambdas
%         lambda = regularization_strength*ones(M+1,C-1);
%         lambda(1,:) = 0; % Don't penalize biases
%         options = [];
%         % Train the model using fminunc
%         fprintf('Training multinomial logistic regression model %d...\n', t);
%         wSoftmax = fminunc(@penalizedL2,zeros((M+1)*(C-1),1), ...
%             options,funObj,lambda(:));
%         % Reshape the predicted weigths
%         wSoftmax = reshape(wSoftmax,[M+1 C-1]);
%         wSoftmax = [wSoftmax zeros(M+1,1)];
%         
%         % Predict the class of the test set using the fitted model
%         % Train
%         [~, y_train_est] = max([ones(length(X_train),1) X_train]*wSoftmax,[],2);
%         ErrorRate_train(k) = sum(y_train ~= y_train_est)/length(y_train);
%         
%         % Test
%         [~, y_test_est] = max([ones(length(X_test),1) X_test]*wSoftmax,[],2);
%         ErrorRate_test(k) = sum(y_test ~= y_test_est)/length(y_test);
        
        
        % ARTIFICIAL NEURAL NETWORKS
        
        % Check the test error of the model with the optimal value of
        % neurons (h) selected on the inner fold.
        ANN = nc_main(X_train_std(:,2:end), y_train+1, X_test_std(:,2:end), y_test+1, h_opt(k));
        y_test_est = ANN.t_est_test;

        ErrorRate_test_ann(k)=sum(y_test ~=y_test_est-1)/length(y_test);

        
        % BASELINE 
        
        % check the errors for the model ithout features
        ErrorRate_train_nofeatures(k)=sum((y_train~=round(mean(y_train))))/length(y_train);
        ErrorRate_test_nofeatures(k)=sum((y_test~=round(mean(y_train))))/length(y_test);


    end
end


%%

    
    [MinErrorRLR, idx_MinErrorRLR]	= min(ErrorRate_test);
    lambda_sel=lambda_opt(idx_MinErrorRLR);
       
    [MinErrorANN, idx_MinErrorANN]	= min(ErrorRate_test_ann);   
    h_sel = h_opt(idx_MinErrorANN);
    
    [MinErrorNoFeatures,idx_MinErrorNoFeatures]	= min(ErrorRate_test_nofeatures);

    fprintf('\n');
    fprintf('Regularized Logistic Regression:\n');
    fprintf('- Training error rate: %4.10f\n', sum(ErrorRate_train)/length(ErrorRate_test));
    fprintf('- Test error rate:     %4.10f\n', sum(ErrorRate_test)/length(ErrorRate_test));
%     fprintf('- R^2 train:           %4.10f\n', (sum(Error_train_nofeatures)-sum(Error_train))/sum(Error_train_nofeatures));
%     fprintf('- R^2 test:            %4.10f\n', (sum(Error_test_nofeatures)-sum(Error_test))/sum(Error_test_nofeatures));   
    fprintf('Artificial Neural Network:\n');
%     fprintf('- Training error rate: %4.10f\n', sum(ErrorRate_train_ann)/length(ErrorRate_test_ann));
    fprintf('- Test error rate:     %4.10f\n', sum(ErrorRate_test_ann)/length(ErrorRate_test_ann));
%     fprintf('- R^2 train:           %4.10f\n', (sum(Error_train_nofeatures)-sum(Error_train))/sum(Error_train_nofeatures));
%     fprintf('- R^2 test:            %4.10f\n', (sum(Error_test_nofeatures)-sum(Error_test))/sum(Error_test_nofeatures));
    fprintf('Baseline:\n');
    fprintf('- Training error rate: %4.10f\n', sum(ErrorRate_train_nofeatures)/length(ErrorRate_test_nofeatures));
    fprintf('- Test error rate:     %4.10f\n', sum(ErrorRate_test_nofeatures)/length(ErrorRate_test_nofeatures));

 table([1:K]',h_opt,ErrorRate_test_ann,lambda_opt,ErrorRate_test,ErrorRate_test_nofeatures)  
%% Logistic Regression Testing
%{
x = 1;
    X_rlr=[ones(size(X_One_Of_K,1),1) X_One_Of_K];
    M_rlr=16;
    attributeNames_rlr={'Offset', attributeNames{1:end}};
    
    lambda = 0.0001;
    
    mu = mean(X_rlr(:,2:end));
    sigma = std(X_rlr(:,2:end));
    
    X_train = X_rlr;
    X_train(:,2:end) = (X_rlr(:,2:end) - mu) ./ sigma;
    
    mdl = fitclinear(X_train, y, ...
                 'Lambda', lambda, ...
                 'Learner', 'logistic', ...
                 'Regularization', 'ridge');
    [y_train_est, p] = predict(mdl, X_train);
    train_error_rate = sum( y ~= y_train_est ) / length(y);



%}

