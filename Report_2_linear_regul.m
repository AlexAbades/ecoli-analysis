%% PART 2

% Regression Part 

%{
% The goal proposed to our dataset is to specify the protein locations
% based in some measurements as signal sequence recognition... 
% For this reason the output "y" is the different categories:

  cp  (cytoplasm)                                    
  im  (inner membrane without signal sequence)       
  pp  (perisplasm)                                   
  imU (inner membrane, uncleavable signal sequence)  
  om  (outer membrane)                               
  omL (outer membrane lipoprotein)  
  imL (inner membrane lipoprotein)                   
  imS (inner membrane, cleavable signal sequence)


Nevertheless, different datasets differ on their specific purpose, for this
specific dataset the goal was to categorize and it has no sense to apply 
lienar regresion for the same output. Therefore, we have rejected the 
origal goal, classifcation. 
With our different features we are now going to predict the score of 
discriminant analysis of the amino acid content of outer membrane and 
periplasmic proteins (aac). We have choosen this feature because it's
continuous and it's a different measurement. lip and chg are binary, alm1
alm2 are the same measurments of ALOM after varying a bit.


  mcg: McGeoch's method for signal sequence recognition.
  gvh: von Heijne's method for signal sequence recognition.
  lip: von Heijne's Signal Peptidase II consensus sequence score.
           Binary attribute.
  chg: Presence of charge on N-terminus of predicted lipoproteins.
	   Binary attribute.
  aac: score of discriminant analysis of the amino acid content of
	   outer membrane and periplasmic proteins.
  alm1: score of the ALOM membrane spanning region prediction program.
  alm2: score of ALOM program after excluding putative cleavable signal
	   regions from the sequence.



load Ecoli_values.mat; 


% Feature engineering: Binarize lip and chg attributes:

NE = array2table(ecoli_norm);
NE.Properties.VariableNames = {'mcg', 'gvh', 'lip', 'chg', 'aac',...
                              'alm1', 'alm2'};

% Get our goal
Y = NE(:,'aac');

% Delete from our data our Y
NE(:,'aac') = [];

% Convert into an array
R_ecoli = table2array(NE);

% Binarize the data that has only two values
R_ecoli = TransformDataset(R_ecoli);

% Reconstruct our data with the target column at the end
R_ecoli = [ R_ecoli table2array(Y)];


figure()
[~,ax]=plotmatrix(R_ecoli); 
set(findall(gcf,'-property','FontSize'),'FontSize',12)
ax(1,1).YLabel.String='mgc'; 
ax(2,1).YLabel.String='gvh'; 
ax(3,1).YLabel.String='lip'; 
ax(4,1).YLabel.String='chg'; 
ax(5,1).YLabel.String='alm1'; 
ax(6,1).YLabel.String='alm2';
ax(7,1).YLabel.String='aac'; 

ax(7,1).XLabel.String='mgc'; 
ax(7,2).XLabel.String='gvh'; 
ax(7,3).XLabel.String='lip';
ax(7,4).XLabel.String='chg'; 
ax(7,5).XLabel.String='alm1';
ax(7,6).XLabel.String='alm2'; 
ax(7,7).XLabel.String='aac'; 

%% Get correlation matrix
corr_R_ecoli = corr(R_ecoli);

% plot matrix plot to see the correlation between attributes
figure()
h = heatmap(corr_R_ecoli); 
h.XDisplayLabels= {'mgc', 'gvh', 'lip', 'chg', 'alm1', 'alm2', 'aac'};
h.YDisplayLabels = {'mgc', 'gvh', 'lip', 'chg', 'alm1', 'alm2', 'aac'};
%}
%% Feature tranformation 
% Our data without standarization or any modification is X
% Drop first column (identification protein) and the last (category column) 
% Get rid also of the binary data, it makes the matrix singular when it try
% to normalize it
clear all; clc;
load Ecoli_values.mat; 

X(:,{'prot_name', 'cat'}) = [];


%% Split our data into the attributes and target 
[X, y, names] = selectTarget(X, 'aac', 0); 
attributeNames = [{'Offset'} names];

%% Regularization 

% add ones to the matrix for the w0 or intercept values 
X=[ones(size(X,1),1) X];
% Select the number of models we are gonna test 
M= size(X,2);

% Selecet the number of folds
K = 10;
% Split dataset into 10 folds
CV = cvpartition(size(X,1), 'Kfold', K);

% Initializate values for lambda 
lambda_tmp=10.^(-5:8);

% Initializate some variables
T = length(lambda_tmp);
Error_train = nan(K,1); % vector to store the training error for each model 
Error_test = nan(K,1); % vector to store the test error for each model

Error_train_rlr = nan(K,1);
Error_test_rlr = nan(K,1);

Error_train_nofeatures = nan(K,1);
Error_test_nofeatures = nan(K,1);

% weights 
w = nan(M,T,K);
lambda_opt = nan(K,1);
w_rlr = nan(M,K);
mu = nan(K, M-1);
sigma = nan(K, M-1);
w_noreg = nan(M,K);

for k = 1:K
    fprintf('Crossvalidation fold %d/%d\n', k, K);
    
    % Extract the training and test set
    X_train = X(CV.training(k), :);
    y_train = y(CV.training(k));
    X_test = X(CV.test(k), :);
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
        
        % Not standarize the binary attributes
        
        mu2 = mean(X_train2(:,2:5));
        sigma2 = std(X_train2(:,2:5));
        X_train2(:,2:5) = (X_train2(:,2:5) - mu2) ./ sigma2+10^-8;
        X_test2(:,2:5) = (X_test2(:,2:5) - mu2) ./ sigma2+10^-8;
      
    
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
        legend(attributeNames(2:end), 'location', 'best');
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

%% Display results
fprintf('\n');
fprintf('Linear regression without feature selection:\n');
fprintf('- Training error: %8.2f\n', sum(Error_train)/sum(CV.TrainSize));
fprintf('- Test error:     %8.2f\n', sum(Error_test)/sum(CV.TestSize));
fprintf('- R^2 train:     %8.2f\n', (sum(Error_train_nofeatures)-sum(Error_train))/sum(Error_train_nofeatures));
fprintf('- R^2 test:     %8.2f\n', (sum(Error_test_nofeatures)-sum(Error_test))/sum(Error_test_nofeatures));
fprintf('Regularized linear regression:\n');
fprintf('- Training error: %8.2f\n', sum(Error_train_rlr)/sum(CV.TrainSize));
fprintf('- Test error:     %8.2f\n', sum(Error_test_rlr)/sum(CV.TestSize));
fprintf('- R^2 train:     %8.2f\n', (sum(Error_train_nofeatures)-sum(Error_train_rlr))/sum(Error_train_nofeatures));
fprintf('- R^2 test:     %8.2f\n', (sum(Error_test_nofeatures)-sum(Error_test_rlr))/sum(Error_test_nofeatures));

fprintf('\n');
fprintf('Weight in last fold: \n');
for m = 1:M
    disp( sprintf(['\t', attributeNames{m},':\t ', num2str(w_rlr(m,end))]))
end
%disp(w_rlr(:,end))






%%



