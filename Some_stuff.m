%%

X = [42, 60, 17, 48, 12];

y = [1, 1, 1, 1, 1];
mu1 = 43;
mu2 = 44;

figure()
plot(X,y, 'r*')
hold on 
plot([43, 44], [1, 1], 'b+');


distance1 = abs(X-mu1);
distance2 = abs(X-mu2); 

ans = distance1 < distance2;
ans2 = distance1 > distance2;
cluster1 = X(ans) 
cluster2 = X(ans2)

mu1 = sum(cluster1)/size(cluster1,2);
mu2 = sum(cluster2)/size(cluster2,2);
mean = mu1 + mu2;
fmean = 0;
count = 0;

K = 10;

N1 = ceil(sqrt(10));
N2 = ceil(K/N1);
for i= 1:K 
    
    distance1 = abs(X-mu1);
    distance2 = abs(X-mu2);
    
    ans1 = distance1 < distance2;
    ans2 = distance1 > distance2;
    cluster1 = X(ans1);
    cluster2 = X(ans2);
    
    mu1 = sum(cluster1)/size(cluster1,2);
    mu2 = sum(cluster2)/size(cluster2,2);
    
    
    subplot(N1, N2, i)
    plot(cluster1, ones(size(cluster1,1)), 'r+')
    hold on
    plot(cluster2, ones(size(cluster2,1)), 'b*')
    hold on 
    plot([mu1, mu2], ones(2, 1), 'o')
    title(fprintf('iteration %u', i))
    count = count + 1;
end
%%




for k = 1:K
    fprintf('Crossvalidation fold %d/%d\n', k, K);
    
    % Extract the training and test set
    X_train = X(CV.training(k), :);
    y_train = y(CV.training(k));
    X_test = X(CV.test(k), :);
    y_test = y(CV.test(k));
    
    
    Xty2 = X_train' * y_train;
    XtX2 = X_train' * X_train;
    
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
        
        % Not standarize the binary attributes 
        mu2 = mean(X_train2(:,2:end));
        mu2(:,5:end) = 0; 
        sigma2 = std(X_train2(:,2:end));
        sigma2(:,5:end) = 0; 
        X_train2(:,2:end) = (X_train2(:,2:end) - mu2) ./ sigma2;
        X_test2(:,2:end) = (X_test2(:,2:end) - mu2) ./ sigma2;
        %%%%%%
        % Should we also standarize the binary attributes??
        %%%%%%
    
        Xty2 = X_train2' * y_train2;
        XtX2 = X_train2' * X_train2;
        for t=1:length(lambda_tmp)   
            % Learn parameter for current value of lambda for the given
            % inner CV_fold
            regularization = lambda_tmp(t) * eye(M);
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
    Xty=(X_train_std'*y_train);
    XtX=X_train_std'*X_train_std;
    
    regularization = lambda_opt(k) * eye(M);
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


%% 

A = 







