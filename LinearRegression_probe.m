close all;
clc;
clear all;
PCA = 0;

load Ecoli_values.mat; 

x_new = y;
y = X_One_Of_K(:,1);
X_One_Of_K = [X_One_Of_K(:,2:end), x_new];

attributeNames = {attributeNames{2:end}, '50K'};
            %%% REGRESSION PART A %%%
%% Explanation of variables 

% what variable is predicted based on which other variables
    
% what you hope to accomplish by the regression.
        
% Mention your feature transformation choices such as one-of-K coding. 
    X_One_Of_K;
% Apply feature transformation such that X has mean 0 and standard deviation 1.
    X_norm;

    %% Regularized Linear regression  - Introduction of lambda

% Estimate regularization error for different values of lambda.
    % Specifically, choose a reasonable range of values of ? (ideally one where the
    % generalization error first drop and then increases), and for each value use
    % K = 10 fold cross-validation to estimate the generalization error.
    
    % include an additional attribute corresponding to the offset
    X_rlr=[ones(size(X_One_Of_K,1),1) X_One_Of_K];
    M_rlr=16;
    attributeNames_rlr={'Offset', attributeNames{1:end}};

    % Crossvalidation
    % Create crossvalidation partition for evaluation of performance of optimal
    % model
    K = 10;
    CV = cvpartition(N, 'Kfold', K);

    % Values of lambda
    lambda_tmp=10.^(-10:10);

    % Initialize variables
    T=length(lambda_tmp);
    Error_train = nan(K,1);
    Error_test = nan(K,1);
    Error_train_rlr = nan(K,1);
    Error_test_rlr = nan(K,1);
    Error_train_nofeatures = nan(K,1);
    Error_test_nofeatures = nan(K,1);
    Error_train2 = nan(T,K);
    Error_test2 = nan(T,K);
    w = nan(M_rlr,T,K);
    lambda_opt = nan(K,1);
    w_rlr = nan(M_rlr,K);
    mu = nan(K, M_rlr-1);
    sigma = nan(K, M_rlr-1);
    w_noreg = nan(M_rlr,K);

    % For each crossvalidation fold
    for k = 1:K
        fprintf('Crossvalidation fold %d/%d\n', k, K);

        % Extract the training and test set
        X_train = X_rlr(CV.training(k), :);
        y_train = y(CV.training(k));
        X_test = X_rlr(CV.test(k), :);
        y_test = y(CV.test(k));

        % Use 10-fold crossvalidation to estimate optimal value of lambda    
        KK = 10;
        CV2 = cvpartition(size(X_train,1), 'Kfold', KK);
        for kk=1:KK
            X_train2 = X_train(CV2.training(kk), :);
            y_train2 = y_train(CV2.training(kk));
            X_test2 = X_train(CV2.test(kk), :);
            y_test2 = y_train(CV2.test(kk));

            % Standardize the training and test set based on training set in
            % the inner fold
            mu2 = mean(X_train2(:,2:end));
            sigma2 = std(X_train2(:,2:end));
            X_train2(:,2:end) = (X_train2(:,2:end) - mu2) ./ sigma2;
            X_test2(:,2:end) = (X_test2(:,2:end) - mu2) ./ sigma2;

            Xty2 = X_train2' * y_train2;
            XtX2 = X_train2' * X_train2;
            for t=1:length(lambda_tmp)   
                % Learn parameter for current value of lambda for the given
                % inner CV_fold
                regularization = lambda_tmp(t) * eye(M_rlr);
                regularization(1,1) = 0; % Remove regularization of bias-term
                w(:,t,kk)=(XtX2+regularization)\Xty2;
                % Evaluate training and test performance
                Error_train2(t,kk) = sum((y_train2-X_train2*w(:,t,kk)).^2);
                Error_test2(t,kk) = sum((y_test2-X_test2*w(:,t,kk)).^2);
            end
        end    

        % Select optimal value of lambda
        [val,ind_opt]=min(sum(Error_test2,2)/sum(CV2.TestSize));
        lambda_opt(k)=lambda_tmp(ind_opt);    

        % Display result for last cross-validation fold (remove if statement to
        % show all folds)
        if k == K
            mfig(sprintf('(%d) Regularized Solution',k));    
            subplot(1,2,1); % Plot error criterion
            semilogx(lambda_tmp, mean(w(2:end,:,:),3),'.-');
            % For a more tidy plot, we omit the attribute names, but you can
            % inspect them using:
            %legend(attributeNames(2:end), 'location', 'best');
            xlabel('\lambda');
            ylabel('Coefficient Values');
            title('Values of w');
            subplot(1,2,2); % Plot error        
            loglog(lambda_tmp,[sum(Error_train2,2)/sum(CV2.TrainSize) sum(Error_test2,2)/sum(CV2.TestSize)],'.-');   
            legend({'Training Error as function of lambda','Test Error as function of lambda'},'Location','SouthEast');
            title(['Optimal value of lambda: 1e' num2str(log10(lambda_opt(k)))]);
            xlabel('\lambda');    
            grid on;
            drawnow;    
        end

        % Standardize datasets in outer fold, and save the mean and standard
        % deviations since they're part of the model (they would be needed for
        % making new predictions)
        mu(k,  :) = mean(X_train(:,2:end));
        sigma(k, :) = std(X_train(:,2:end));

        X_train_std = X_train;
        X_test_std = X_test;
        X_train_std(:,2:end) = (X_train(:,2:end) - mu(k , :)) ./ sigma(k, :);
        X_test_std(:,2:end) = (X_test(:,2:end) - mu(k, :)) ./ sigma(k, :);

        % Estimate w for the optimal value of lambda
        Xty= (X_train_std'*y_train)
        XtX= X_train_std'*X_train_std

        regularization = lambda_opt(k) * eye(M_rlr);
        regularization(1,1) = 0; 
        w_rlr(:,k) = (XtX+regularization)\Xty;

        % evaluate training and test error performance for optimal selected value of
        % lambda
        Error_train_rlr(k) = sum((y_train-X_train_std*w_rlr(:,k)).^2);
        Error_test_rlr(k) = sum((y_test-X_test_std*w_rlr(:,k)).^2);

        % Compute squared error without regularization
        w_noreg(:,k)=XtX\Xty;
        Error_train(k) = sum((y_train-X_train_std*w_noreg(:,k)).^2);
        Error_test(k) = sum((y_test-X_test_std*w_noreg(:,k)).^2);

        % Compute squared error without using the input data at all
        Error_train_nofeatures(k) = sum((y_train-mean(y_train)).^2);
        Error_test_nofeatures(k) = sum((y_test-mean(y_train)).^2);

    end
    
    lambda_opt(k)=lambda_tmp(ind_opt);
  
  [MinErrorNoFeatures,idx_MinErrorNoFeatures]   = min(Error_test_nofeatures);
  [MinErrorRLR, idx_MinErrorRLR]                = min(Error_test_rlr);
  [MinErrorNoReg, idx_MinErrorNoReg]            = min(Error_test);

  lambda_sel=lambda_opt(idx_MinErrorRLR);
% Include a figure of the estimated generalization error as a function of ? in the
% report and briefly discuss the result.
   
    mfig(sprintf('Regularized Solution',idx_MinErrorRLR));          
        loglog(lambda_tmp,sum(Error_test2,2)/sum(CV2.TestSize),'.-');   
        legend({'Test Error as function of lambda'},'Location','SouthEast');
        title(['Optimal value of lambda: 1e' num2str(log10(lambda_opt(idx_MinErrorRLR)))]);
        xlabel('\lambda'); 
        ylabel('Generalization Error');
        grid on;
        drawnow;    

    % Display results
    fprintf('\n');
    fprintf('Linear regression without feature selection:\n');
    fprintf('- Training error: %4.10f\n', sum(Error_train)/sum(CV.TrainSize));
    fprintf('- Test error:     %4.10f\n', sum(Error_test)/sum(CV.TestSize));
    fprintf('- R^2 train:      %4.10f\n', (sum(Error_train_nofeatures)-sum(Error_train))/sum(Error_train_nofeatures));
    fprintf('- R^2 test:       %4.10f\n', (sum(Error_test_nofeatures)-sum(Error_test))/sum(Error_test_nofeatures));
    fprintf('Regularized linear regression:\n');
    fprintf('- Training error: %4.10f\n', sum(Error_train_rlr)/sum(CV.TrainSize));
    fprintf('- Test error:     %4.10f\n', sum(Error_test_rlr)/sum(CV.TestSize));
    fprintf('- R^2 train:      %4.10f\n', (sum(Error_train_nofeatures)-sum(Error_train_rlr))/sum(Error_train_nofeatures));
    fprintf('- R^2 test:       %4.10f\n', (sum(Error_test_nofeatures)-sum(Error_test_rlr))/sum(Error_test_nofeatures));

%     fprintf('\n');
%     fprintf('Weight in last fold: \n');
%     for m = 1:M_rlr
%         disp( sprintf(['\t', attributeNames_rlr{m},':\t ', num2str(w_rlr(m,end))]))
%     end
    %disp(w_rlr(:,end))
    
    
    %% Reg Part B

% Estimate regularization error for different values of lambda.
    % Specifically, choose a reasonable range of values of ? (ideally one where the
    % generalization error first drop and then increases), and for each value use
    % K = 10 fold cross-validation to estimate the generalization error.
    tic
    % include an additional attribute corresponding to the offset
    X_rlr=[ones(size(X_One_Of_K,1),1) X_One_Of_K];
    M_rlr=16;
    attributeNames_rlr={'Offset', attributeNames{1:end}};

    % Crossvalidation
    % Create crossvalidation partition for evaluation of performance of optimal
    % model
    K = 2;
    KK = 2;
    CV = cvpartition(N, 'Kfold', K);

    NTrain = 1; % Number of re-trains of neural network
    bestnet=cell(K,1);
    lambda_tmp=10.^(-2:3);
    h_tmp = [10 50 100 150];
    
    % Initialize variables
    T=length(lambda_tmp);
    Error_train_ann = nan(K,1);
    Error_test_ann = nan(K,1);
    Error_train_ann2 = nan(K,1);
    Error_test_ann2 = nan(K,1);
    Error_train_rlr = nan(K,1);
    Error_test_rlr = nan(K,1);
    Error_train_nofeatures = nan(K,1);
    Error_test_nofeatures = nan(K,1);
    Error_train2 = nan(T,K);
    Error_test2 = nan(T,K);
    w = nan(M_rlr,T,K);
    lambda_opt = nan(K,1);
    h_opt = nan(K,1);
    h_opt2 = nan(KK,1);
    w_rlr = nan(M_rlr,K);
    mu = nan(K, M_rlr-1);
    sigma = nan(K, M_rlr-1);
    w_noreg = nan(M_rlr,K);
%     ANN = nan(K,1);
    
    
    
    % For each crossvalidation fold
    for k = 1:K
        fprintf('Crossvalidation fold %d/%d\n', k, K);

        % Extract the training and test set
        X_train = X_rlr(CV.training(k), :);
        y_train = y(CV.training(k));
        X_test = X_rlr(CV.test(k), :);
        y_test = y(CV.test(k));
        
        % Use 10-fold crossvalidation to estimate optimal value of lambda    
%         KK = 10;
        CV2 = cvpartition(size(X_train,1), 'Kfold', KK);
        for kk=1:KK
            fprintf('Inner fold %d/%d\n', kk, KK);
            X_train2 = X_train(CV2.training(kk), :);
            y_train2 = y_train(CV2.training(kk));
            X_test2 = X_train(CV2.test(kk), :);
            y_test2 = y_train(CV2.test(kk));

            % Standardize the training and test set based on training set in
            % the inner fold
            mu2 = mean(X_train2(:,2:end));
            sigma2 = std(X_train2(:,2:end));
            X_train2(:,2:end) = (X_train2(:,2:end) - mu2) ./ sigma2;
            X_test2(:,2:end) = (X_test2(:,2:end) - mu2) ./ sigma2;

            Xty2 = X_train2' * y_train2;
            XtX2 = X_train2' * X_train2;
%             
            for t=1:length(lambda_tmp)   
                % Learn parameter for current value of lambda for the given
                % inner CV_fold
                regularization = lambda_tmp(t) * eye(M_rlr);
                regularization(1,1) = 0; % Remove regularization of bias-term
                w(:,t,kk)=(XtX2+regularization)\Xty2;
                % Evaluate training and test performance
                Error_train2(t,kk) = sum((y_train2-X_train2*w(:,t,kk)).^2);
                Error_test2(t,kk) = sum((y_test2-X_test2*w(:,t,kk)).^2);
            end
            
            MSEBest = inf;
            for t = 1:length(h_tmp)
                fprintf('ANN: %d/%d\n', t, length(h_tmp));
                netwrk = nr_main(X_train2(:,2:end), y_train2, X_test2(:,2:end), y_test2, h_tmp(t));
                if netwrk.mse_train(end)<MSEBest, bestnet{kk} = netwrk; MSEBest=netwrk.mse_train(end); MSEBest=netwrk.mse_train(end); h_opt2(kk) = h_tmp(t); end            
            end
            % Predict model on test and training data    
            y_train_est = bestnet{kk}.t_pred_train;    
            y_test_est = bestnet{kk}.t_pred_test;        

            % Compute least squares error
            Error_train_ann2(kk) = sum((y_train2-y_train_est).^2);
            Error_test_ann2(kk) = sum((y_test2-y_test_est).^2); 
            
        end    

        % Select optimal value of lambda
        [val,ind_opt]=min(sum(Error_test2,2)/sum(CV2.TestSize));
        lambda_opt(k)=lambda_tmp(ind_opt);  
        
        % Select optimal value of lambda
        [val,ind_opt]=min(sum(Error_test_ann2,2)/sum(CV2.TestSize));
        h_opt(k)=h_opt2(ind_opt);
        ANN{k} = bestnet{ind_opt};
        

        % Display result for last cross-validation fold (remove if statement to
        % show all folds)
        if k == K
            mfig(sprintf('(%d) Regularized Solution',k));    
            subplot(1,2,1); % Plot error criterion
            semilogx(lambda_tmp, mean(w(2:end,:,:),3),'.-');
            % For a more tidy plot, we omit the attribute names, but you can
            % inspect them using:
            %legend(attributeNames(2:end), 'location', 'best');
            xlabel('\lambda');
            ylabel('Coefficient Values');
            title('Values of w');
            subplot(1,2,2); % Plot error        
            loglog(lambda_tmp,[sum(Error_train2,2)/sum(CV2.TrainSize) sum(Error_test2,2)/sum(CV2.TestSize)],'.-');   
            legend({'Training Error as function of lambda','Test Error as function of lambda'},'Location','SouthEast');
            title(['Optimal value of lambda: 1e' num2str(log10(lambda_opt(k)))]);
            xlabel('\lambda');    
            grid on;
            drawnow;    
        end

        % Standardize datasets in outer fold, and save the mean and standard
        % deviations since they're part of the model (they would be needed for
        % making new predictions)
        mu(k,  :) = mean(X_train(:,2:end));
        sigma(k, :) = std(X_train(:,2:end));

        X_train_std = X_train;
        X_test_std = X_test;
        X_train_std(:,2:end) = (X_train(:,2:end) - mu(k , :)) ./ sigma(k, :);
        X_test_std(:,2:end) = (X_test(:,2:end) - mu(k, :)) ./ sigma(k, :);

        % Estimate w for the optimal value of lambda
        Xty= (X_train_std'*y_train);
        XtX= X_train_std'*X_train_std;

        regularization = lambda_opt(k) * eye(M_rlr);
        regularization(1,1) = 0; 
        w_rlr(:,k) = (XtX+regularization)\Xty;

        % evaluate training and test error performance for optimal selected value of
        % lambda
        Error_train_rlr(k) = sum((y_train-X_train_std*w_rlr(:,k)).^2);
        Error_test_rlr(k) = sum((y_test-X_test_std*w_rlr(:,k)).^2);

        % Compute squared error without regularization
        w_noreg(:,k)=XtX\Xty;
        Error_train(k) = sum((y_train-X_train_std*w_noreg(:,k)).^2);
        Error_test(k) = sum((y_test-X_test_std*w_noreg(:,k)).^2);

        % Compute squared error without using the input data at all
        Error_train_nofeatures(k) = sum((y_train-mean(y_train)).^2);
        Error_test_nofeatures(k) = sum((y_test-mean(y_train)).^2);
        
        % Predict model on test and training data        
%             y_test_est = ANN{1,k}.Wo*(ANN{1,k}.Wi*X_test');        
%             y_train_est = ANN{1,k}.Wo*(ANN{1,k}.Wi*X_train');
            y_train_est = nr_eval(ANN{k},X_train(:,2:end));    
            y_test_est = nr_eval(ANN{k},X_test(:,2:end));

            % Compute least squares error
            Error_train_ann(k) = sum((y_train-y_train_est).^2);
            Error_test_ann(k) = sum((y_test-y_test_est).^2); 

    end
    
    lambda_opt(k)=lambda_tmp(ind_opt);
  
  [MinErrorNoFeatures,idx_MinErrorNoFeatures]   = min(Error_test_nofeatures);
  [MinErrorRLR, idx_MinErrorRLR]                = min(Error_test_rlr);
  [MinErrorANN, idx_MinErrorANN]            = min(Error_test_ann);

  lambda_sel=lambda_opt(idx_MinErrorRLR);
  h_sel = h_opt(idx_MinErrorANN);
% Include a figure of the estimated generalization error as a function of ? in the
% report and briefly discuss the result.
toc   
    mfig(sprintf('Regularized Solution',idx_MinErrorRLR));          
        loglog(lambda_tmp,sum(Error_test2,2)/sum(CV2.TestSize),'.-');   
        legend({'Test Error as function of lambda'},'Location','SouthEast');
        title(['Optimal value of lambda: 1e' num2str(log10(lambda_opt(idx_MinErrorRLR)))]);
        xlabel('\lambda'); 
        ylabel('Generalization Error');
        grid on;
        drawnow;    

    % Display results
    fprintf('\n');
    fprintf('Linear regression without feature selection:\n');
    fprintf('- Training error: %4.10f\n', sum(Error_train_ann)/sum(CV.TrainSize));
    fprintf('- Test error:     %4.10f\n', sum(Error_test_ann)/sum(CV.TestSize));
    fprintf('- R^2 train:      %4.10f\n', (sum(Error_train_nofeatures)-sum(Error_train_ann))/sum(Error_train_nofeatures));
    fprintf('- R^2 test:       %4.10f\n', (sum(Error_test_nofeatures)-sum(Error_test_ann))/sum(Error_test_nofeatures));
    fprintf('Regularized linear regression:\n');
    fprintf('- Training error: %4.10f\n', sum(Error_train_rlr)/sum(CV.TrainSize));
    fprintf('- Test error:     %4.10f\n', sum(Error_test_rlr)/sum(CV.TestSize));
    fprintf('- R^2 train:      %4.10f\n', (sum(Error_train_nofeatures)-sum(Error_train_rlr))/sum(Error_train_nofeatures));
    fprintf('- R^2 test:       %4.10f\n', (sum(Error_test_nofeatures)-sum(Error_test_rlr))/sum(Error_test_nofeatures));
    ANN_Et = Error_test_ann/CV.TestSize
    RLR_Et = Error_test_rlr./CV.TestSize
    Base_Et = Error_test_nofeatures./CV.TestSize
%     fprintf('\n');
%     fprintf('Weight in last fold: \n');
%     for m = 1:M_rlr
%         disp( sprintf(['\t', attributeNames_rlr{m},':\t ', num2str(w_rlr(m,end))]))
%     end
    %disp(w_rlr(:,end))

% %%
% 
% % K-fold crossvalidation
% K = 10;
% CV = cvpartition(N,'Kfold', K);
% 
% % Parameters for neural network classifier
% NHiddenUnits = 1;  % Number of hidden units
% NTrain = 1; % Number of re-trains of neural network
% 
% % Variable for regression error
% Error_train = nan(K,1);
% Error_test = nan(K,1);
% Error_train_nofeatures = nan(K,1);
% Error_test_nofeatures = nan(K,1);
% bestnet=cell(K,1);
% 
% for k = 1:K % For each crossvalidation fold
%     fprintf('Crossvalidation fold %d/%d\n', k, CV.NumTestSets);
% 
%     % Extract training and test set
%     X_train = X_One_Of_K(CV.training(k), :);
%     y_train = y(CV.training(k));
%     X_test = X_One_Of_K(CV.test(k), :);
%     y_test = y(CV.test(k));
% 
%     % Fit neural network to training set
%     MSEBest = inf;
%     for t = 1:NTrain
%         netwrk = nr_main(X_train, y_train, X_test, y_test, NHiddenUnits);
%         if netwrk.mse_train(end)<MSEBest, bestnet{k} = netwrk; MSEBest=netwrk.mse_train(end); MSEBest=netwrk.mse_train(end); end
%     end
%     
%     % Predict model on test and training data    
%     y_train_est = bestnet{k}.t_pred_train;    
%     y_test_est = bestnet{k}.t_pred_test;        
%     
%     % Compute least squares error
%     Error_train(k) = sum((y_train-y_train_est).^2);
%     Error_test(k) = sum((y_test-y_test_est).^2); 
%         
%     % Compute least squares error when predicting output to be mean of
%     % training data
%     Error_train_nofeatures(k) = sum((y_train-mean(y_train)).^2);
%     Error_test_nofeatures(k) = sum((y_test-mean(y_train)).^2);            
% end
% 
% % Print the least squares errors
% %% Display results
% % clc
% fprintf('\n');
% fprintf('Neural network regression without feature selection:\n');
% fprintf('- Training error: %8.2f\n', sum(Error_train)/sum(CV.TrainSize));
% fprintf('- Test error:     %8.2f\n', sum(Error_test)/sum(CV.TestSize));
% fprintf('- R^2 train:     %8.2f\n', (sum(Error_train_nofeatures)-sum(Error_train))/sum(Error_train_nofeatures));
% fprintf('- R^2 test:     %8.2f\n', (sum(Error_test_nofeatures)-sum(Error_test))/sum(Error_test_nofeatures));
% 
% 
% % Display the trained network 
% mfig('Trained Network');
% k=1; % cross-validation fold
% displayNetworkRegression(bestnet{k});
% 
% % Display how network predicts (only for when there are two attributes)
% if size(X_train,2)==2 % Works only for problems with two attributes
% 	mfig('Decision Boundary');
% 	displayDecisionFunctionNetworkRegression(X_train, y_train, X_test, y_test, bestnet{k});
% end

%% Display results

fprintf('\n');
fprintf('Neural network regression without feature selection:\n');
fprintf('- Training error: %8.2f\n', sum(Error_train_ann)/sum(CV.TrainSize));
fprintf('- Test error:     %8.2f\n', sum(Error_test_ann)/sum(CV.TestSize));
fprintf('- R^2 train:     %8.2f\n', (sum(Error_train_nofeatures)-sum(Error_train_ann))/sum(Error_train_nofeatures));
fprintf('- R^2 test:     %8.2f\n', (sum(Error_test_nofeatures)-sum(Error_test_ann))/sum(Error_test_nofeatures));


% Display the trained network 
mfig('Trained Network');
k=1; % cross-validation fold
displayNetworkRegression(bestnet{k});

% Display how network predicts (only for when there are two attributes)
if size(X_train,2)==2 % Works only for problems with two attributes
	mfig('Decision Boundary');
	displayDecisionFunctionNetworkRegression(X_train(:,2:end), y_train, X_test(:,2:end), y_test, bestnet{k});
    hold off
end
