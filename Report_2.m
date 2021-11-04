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

%}

load Ecoli_values.mat;
 

% Feature engineering: Binarize lip and chg attributes:

NE = array2table(ecoli_norm);
NE.Properties.VariableNames = {'mcg', 'gvh', 'lip', 'chg', 'aac',...
                              'alm1', 'alm2'};

% Get our goal
Y = NE(:,'aac');

% Delete from our data our Y
NE(:,'aac') = [];

R_ecoli = table2array(NE);

R_ecoli = TransformDataset(R_ecoli);

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
holf off
%%
corr_R_ecoli = corr(R_ecoli);
figure()
h = heatmap(corr_R_ecoli); 
h.XDisplayLabels= {'mgc', 'gvh', 'lip', 'chg', 'alm1', 'alm2', 'aac'};
h.YDisplayLabels = {'mgc', 'gvh', 'lip', 'chg', 'alm1', 'alm2', 'aac'};


%% Regularization 

% add ones to the matrix for the w0 or intercept values 
X=[ones(size(X,1),1) R_ecoli];
% Select the number of models we are gonna test 
M= size(X,1);

% Selecet the number of folds
K = 10;
% Split dataset 
CV = cvpartition(size(X,1), 'Kfold', K);


% Initializate values for lambda 
lambda_tmp=10.^(-5:8);

% Initializate some variables

T=length(lambda_tmp);
Error_train = nan(K,1); % vector to store the training error for each model 
Error_test = nan(K,1); % vector to store the test error for each model

Error_train_rlr = nan(K,1);
Error_test_rlr = nan(K,1);

Error_train_nofeatures = nan(K,1);
Error_test_nofeatures = nan(K,1);

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
        mu2 = mean(X_train2(:,2:end));
        sigma2 = std(X_train2(:,2:end));
        X_train2(:,2:end) = (X_train2(:,2:end) - mu2) ./ sigma2;
        X_test2(:,2:end) = (X_test2(:,2:end) - mu2) ./ sigma2;
        
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














