# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 09:59:53 2021

@author: G531
"""




# exercise 8.3.3 Fit regularized multinomial regression
import matplotlib.pyplot as plt
from scipy.io import loadmat
from toolbox_02450 import dbplotf, train_neural_net, visualize_decision_boundary
import numpy as np
import sklearn.linear_model as lm
from sklearn import model_selection
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from sklearn.linear_model import LogisticRegression
import torch
from sklearn import model_selection
from toolbox_02450 import train_neural_net, draw_neural_net
from scipy import stats
from sklearn.dummy import DummyClassifier

# Load Matlab data file and extract variables of interest
mat_data = loadmat(r'C:\Users\G531\Documents\8 - Github\ecoli-analysis/Ecoli_python.mat')



X = mat_data['X']
X_train = mat_data['X_train']
X_test = mat_data['X_test']
y = mat_data['y'].squeeze()
y_train = mat_data['y_train'].squeeze()
y_test = mat_data['y_test'].squeeze()

attributeNames = [name[0] for name in mat_data['attributeNames'].squeeze()]
classNames = [name[0][0] for name in mat_data['classNames']]

# Find the unabalanced classes and erase them 
category2 = np.where(y == 2)
category3 = np.where(y == 3)
cate = np.concatenate((category2[0], category3[0]), axis=None)
X = np.delete(X, cate, axis=0)
y = np.delete(y, cate, axis=0)

# select the categories above 3 
ind = y > 3
y[ind] = y[ind]-2

N, M = X.shape # M-> Number of attributes, N-> Number of instances 
C = len(set(y)) # Number of categories we have to predict  



# =============================================================================
# 
#               PARAMETERS FOR CORSS VALIDATION AND MODEL

# Number of outer fold folders.
K = 5
# # Number of inner fold folders.
KK = 5

# Multinomial parameters 
# Check the output 10**-n
# Minimum Value 
min_lambda = -3
# Maximum Value 
max_lambda = 5

# Net parameters 
# Parameters for neural network classifier
n_hidden_units = np.array([2, 4])   # number of hidden units
n_replicates = 1                    # number of networks trained in each k-fold
max_iter = 10000                    # stop criterion 2 (max epochs in training)
#
#
# =============================================================================



# Values of lambda for multiclass 
lambdas = np.power(10.,range(min_lambda,max_lambda))

test_error_rate = np.array([])
Error_train_rate = np.empty((K,1))

# Error in the inner loop
train_error_rate_inner = np.empty((len(lambdas),KK))
test_error_rate_inner = np.empty((len(lambdas),KK))

# Error outer loops 
test_error_rate_outer = np.empty((K,1))
train_error_rate_outer = np.empty((K,1))

# Dummy 
test_error_rate_dummy_outer = np.empty((K,1))

# Variables to sotre the 
prediction_inner = []
y_test_inner_store = []
prediction_outer = []
y_test_outer_store = []

optimal_lambdas = np.empty((K,1))

# Split the data into K folds 
# CV = model_selection.KFold(K, shuffle=True)
skf_outer = StratifiedKFold(n_splits=K)
skf_outer.get_n_splits(X, y)

# Number of lambdas that we are going to test 
T = len(lambdas)
# Attributes weights outer loop 
w_est_inner_tot = np.empty((KK,T,M))
w_est_inner = np.empty((T,M))
coefficient_norm_inner = np.empty((T,KK))
# Attributes weights outer loop
w_est_outer = np.empty((K,M))
coefficient_norm_outer = np.zeros(K)


# ARTIFICIAL NEURAL NETWORK

optimal_neurons = np.empty((K,1))

# Initialize Errors
# Outer loop error 
train_error_rate_inner_ann = np.empty((len(n_hidden_units),KK))

# Inner loop error
test_error_rate_ann_outer = np.empty((K,1))
train_error_rate_ann_outer = np.empty((K,1))

# Make figure for holding summaries (errors and learning curves)
summaries, summaries_axes = plt.subplots(1,2, figsize=(10,5))

# Make a list for storing assigned color of learning curve for up to K=10
color_list = ['tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:pink',
              'tab:gray', 'tab:olive', 'tab:cyan', 'tab:red', 'tab:blue']


#%% Model fitting and prediction
# Standardize data based on training set
k = 0 
# Outer loop 
for train_index_outer, test_index_outer in skf_outer.split(X, y):
    print('\nOuter fold: ',k, '/', K, '\n')
    # Extract training and test set for current CV fold
    X_train_outer = X[train_index_outer]
    y_train_outer = y[train_index_outer]
    X_test_outer = X[test_index_outer]
    y_test_outer = y[test_index_outer]

    # Initialise internal cross validation split
    internal_cross_validation = KK   
    skf_inner = StratifiedKFold(n_splits=internal_cross_validation)
    skf_inner.get_n_splits(X_train_outer)

    kk = 0
    # Inner loop 
    for train_index_inner, test_index_inner in skf_inner.split(X_train_outer, y_train_outer):
        print('Inner fold: ', kk, '/', internal_cross_validation)
        # Extract training and test set for current CV fold
        X_train_inner = X_train_outer[train_index_inner]
        y_train_inner = y_train_outer[train_index_inner]
        X_test_inner = X_train_outer[test_index_inner]
        y_test_inner = y_train_outer[test_index_inner]
        
        # Normalise the data based on the training set
        mu_inner = np.mean(X_train_inner, 0)
        sigma_inner = np.std(X_train_inner, 0)
        X_train_inner = (X_train_inner - mu_inner) / sigma_inner
        X_test_inner = (X_test_inner - mu_inner) / sigma_inner
    
# =============================================================================
#       # MULTIVARIATE LOGISTIC REGRESSION model for each lambda
# =============================================================================
        for lamb in lambdas:
            print('Testing de value of lambda: ', lamb)
            # Specify the model
            mdl_inner = LogisticRegression(multi_class='multinomial',
                                      penalty='l2', random_state=(1),
                                      C=1/lamb, 
                                      max_iter= 5000)
            # Fit the model 
            mdl_inner.fit(X_train_inner,y_train_inner)
            # Estimate the classes 
            y_train_est_inner = mdl_inner.predict(X_train_inner)
            y_test_est_inner = mdl_inner.predict(X_test_inner)
            # Store the predictions 
            prediction_inner.append(y_test_est_inner)
            y_test_inner_store.append(y_test_inner)
            # Estimate the error of the model for each lambda and inner fold 
            train_error_rate_inner[lambdas==lamb, kk] = np.sum(y_train_est_inner!=y_train_inner) / len(y_train_inner)
            test_error_rate_inner[lambdas==lamb,kk] = np.sum(y_test_est_inner!=y_test_inner) / len(y_test_inner)
            # Get the weigths of the attributes
            w_est_inner[lambdas==lamb,:] = mdl_inner.coef_[0] 
            w_est_inner_tot[kk,lambdas==lamb,:] = mdl_inner.coef_[0] 
            coefficient_norm_inner[lambdas==lamb,kk] = np.sqrt(np.sum(w_est_inner**2))
        
# =============================================================================
#       # ARTIFICIAL NEURAL NETWORK 
# =============================================================================
        
        # Transofrm the data into tensor
        X_train_inner_ann = torch.Tensor(X_train_inner)
        y_train_inner_ann = torch.Tensor(y_train_inner)
        # Transform the data into a long scalar by petition
        y_train_inner_ann = y_train_inner_ann.type(torch.LongTensor)
        X_test_inner_ann = torch.Tensor(X_test_inner)
        y_test_inner_ann = torch.Tensor(y_test_inner)
        

        for neuron in n_hidden_units:
            print('Creating and testing model for Nº neurons: ', neuron,)
            # Create a model with different neurons
            model = lambda: torch.nn.Sequential(
                            torch.nn.Linear(M, neuron), #M features to H hiden units
                            # 1st transfer function, either Tanh or ReLU:
                            torch.nn.ReLU(),                            #torch.nn.ReLU(),
                            torch.nn.Linear(neuron, C), # H hidden units to C output neurons (Theoretically the number of classes)
                            torch.nn.Softmax(dim=1) # final tranfer function
                            )
            loss_fn = torch.nn.CrossEntropyLoss()
            
            # Train the net on training data
            net, final_loss, learning_curve = train_neural_net(model,
                                                        loss_fn,
                                                        X=X_train_inner_ann,
                                                        y=y_train_inner_ann,
                                                        n_replicates=n_replicates,
                                                        max_iter=max_iter)
            
            # Predictions of the neural network (From function help)
            y_test_est_inner_ann = net(X_test_inner_ann)
            y_class = torch.max(y_test_est_inner_ann, dim=1)[1]
            # Transform to integer type, (not necessary if we create another datafor the ann)
            y_test_inner_ann = y_test_inner_ann.type(dtype=torch.uint8)
            # Compare the predicted values of the ann with the original 
            e = y_class != y_test_inner_ann
            e_rate = (sum(e).type(torch.float)/len(y_test)).data.numpy()
            train_error_rate_inner_ann[n_hidden_units==neuron, kk] = float(e_rate)

        
        # counter inner loop     
        kk =+ 1
# =============================================================================
#   # MULTINOMIAL LOGISTIC REGRESSION EVALUATION  
# =============================================================================
    # Average over the error of all the models (approximation) Should we  
    y_test_est_inner_mean = np.mean(test_error_rate_inner,1)
    # Select the minimum error
    optimal_lambdas[k] = lambdas[y_test_est_inner_mean==min(y_test_est_inner_mean)][0]
    
    # Standarize the data 
    mu_outer = np.mean(X_train_outer, 0)
    sigma_outer = np.std(X_train_outer, 0)
    X_train_outer = (X_train_outer - mu_outer) / sigma_outer
    X_test_outer = (X_test_outer - mu_outer) / sigma_outer
    
    
    # Build the model for the optimal lambda
    # Train Optimal model with lambda calculated in the inner loop and compute the generalised error
    mdl_outer = LogisticRegression(multi_class='multinomial',
                                      penalty='l2', random_state=(1),
                                      C=1/optimal_lambdas[k][0], 
                                      max_iter= 5000)

    # Fit the model 
    mdl_outer.fit(X_train_outer,y_train_outer)
    # Estimate the classes 
    y_train_est_outer = mdl_outer.predict(X_train_outer)
    y_test_est_outer = mdl_outer.predict(X_test_outer)
    # Store the predictions 
    prediction_outer.append(y_test_est_outer)
    y_test_outer_store.append(y_test_outer)
    # Estimate the error of the model for each lambda and inner fold 
    train_error_rate_outer[k] = np.sum(y_train_est_outer!=y_train_outer) / len(y_train_outer)
    test_error_rate_outer[k] = np.sum(y_test_est_outer!=y_test_outer) / len(y_test_outer)    
    # Get attributes
    w_est_outer[k,:] = mdl_outer.coef_[0] 
    coefficient_norm_outer[k] = np.sqrt(np.sum(w_est_outer**2))
    
    
    
# =============================================================================
#     # ARTIFICIAL NEURONAL NETWORK EVALUATION
# =============================================================================
    # Average over the error of all the models (approximation because of the mean)   
    y_test_est_inner_ann_mean = np.mean(train_error_rate_inner_ann,1)
    # Select the minimum error
    neuron_index = y_test_est_inner_ann_mean==min(y_test_est_inner_ann_mean)
    
    if neuron_index.all():
        print('Both equal')
        optimal_neurons[k] = y_test_est_inner_ann_mean[0]
    else:
        optimal_neurons[k] = n_hidden_units[y_test_est_inner_ann_mean==min(y_test_est_inner_ann_mean)][0]
    
    X_train_outer_ann = torch.Tensor(X_train_outer)
    y_train_outer_ann = torch.Tensor(y_train_outer)
    # Transform the data into a long scalar by petition
    y_train_outer_ann = y_train_outer_ann.type(torch.LongTensor)
    X_test_outer_ann = torch.Tensor(X_test_outer)
    y_test_outer_ann = torch.Tensor(y_test_outer)
    
    model = lambda: torch.nn.Sequential(
                            torch.nn.Linear(M, neuron), #M features to H hiden units
                            # 1st transfer function, either Tanh or ReLU:
                            torch.nn.ReLU(),                            #torch.nn.ReLU(),
                            torch.nn.Linear(neuron, C), # H hidden units to C output neurons (Theoretically the number of classes)
                            torch.nn.Softmax(dim=1) # final tranfer function
                            )
    loss_fn = torch.nn.CrossEntropyLoss()
            
    # Train the net on training data
    net, final_loss, learning_curve = train_neural_net(model,
                                                        loss_fn,
                                                        X=X_train_inner_ann,
                                                        y=y_train_inner_ann,
                                                        n_replicates=n_replicates,
                                                        max_iter=max_iter)
            
    # Predictions of the neural network (From function help)
    y_test_est_inner_ann = net(X_test_inner_ann)
    y_class = torch.max(y_test_est_inner_ann, dim=1)[1]
    # Transform to integer type, (not necessary if we create another datafor the ann)
    y_test_inner_ann = y_test_inner_ann.type(dtype=torch.uint8)
    # Compare the predicted values of the ann with the original 
    e = y_class != y_test_inner_ann
    e_rate = (sum(e).type(torch.float)/len(y_test)).data.numpy()
    test_error_rate_ann_outer[k] = e_rate
    
    
# =============================================================================
#     # DISPLAY DE LEARNING CURVES 
# =============================================================================
     # Display the learning curve for the best net in the current fold
    h, = summaries_axes[0].plot(learning_curve, color=color_list[k])
    h.set_label('CV fold {0}'.format(k+1))
    summaries_axes[0].set_xlabel('Iterations')
    summaries_axes[0].set_xlim((0, max_iter))
    summaries_axes[0].set_ylabel('Loss')
    summaries_axes[0].set_title('Learning curves')
    
# =============================================================================
#     # DUMMY MODEL
# =============================================================================
    # calculate the the probability of being in a class by the most freq 
    dummy_clf = DummyClassifier(strategy="most_frequent")
    dummy_clf.fit(X_train_outer, y_train_outer)
    y_test_est_dummy_outer = dummy_clf.predict(X_test_outer)
    test_error_rate_dummy_outer[k] = np.sum(y_test_est_dummy_outer!=y_test_outer) / len(y_test_outer)    
    
    
    
    # counter outer loop 
    k += 1    





# =============================================================================
#
# 
#                             OUT OF THE LOOP 
# 
# 
# =============================================================================


# =============================================================================
# # CREATE A TABLE
# =============================================================================
data = np.array(list(range(1,K+1)))
data = np.vstack((data, optimal_lambdas.T,test_error_rate_outer.T*100,
           optimal_neurons.T,test_error_rate_ann_outer.T*100,
           test_error_rate_dummy_outer.T*100))

arrays = [
        np.array(["Outer fold", "ANN", "ANN", "Multi LR", "Multi LR", "BaseLine"]),
        np.array(["Folder", "Optimal NeuronS", "Test Error", "Optimal Lambdas", "dfTest Error", "Test Error"]),
        ]
tuples = list(zip(*arrays))

index = pd.MultiIndex.from_tuples(tuples, names=["first", "second"])

df = pd.DataFrame(data=data, index=arrays).T
        

# =============================================================================
# # CREATE SOME PLOTS
# =============================================================================


plt.figure(figsize=(8,8))
#plt.plot(np.log10(lambda_interval), train_error_rate*100)
#plt.plot(np.log10(lambda_interval), test_error_rate*100)
#plt.plot(np.log10(opt_lambda), min_error*100, 'o')
plt.semilogx(lambdas, train_error_rate_inner[:,1]*100)
plt.semilogx(lambdas, test_error_rate_inner[:,1]*100)
p = lambdas[test_error_rate_inner[:,1] == min(test_error_rate_inner[:,1])]
plt.semilogx(lambdas[test_error_rate_inner[:,1] == min(test_error_rate_inner[:,1])][0],
             min(test_error_rate_inner[:,1])*100, 'o')
# plt.text(1e-8, 3, "Minimum test error: " + str(np.round(min(test_error_rate_inner[:,1])*100,2)) + ' % at 1e' +  str(np.round(np.log10(optimal_lambdas[0]),2)))
plt.xlabel('Regularization strength, $\log_{10}(\lambda)$')
plt.ylabel('Error rate (%)')
plt.title('Classification error')
plt.legend(['Training error','Test error','Test minimum'],loc='upper right')
# plt.ylim([0, 4])
plt.grid()
plt.show()    

plt.figure(figsize=(8,8))
plt.semilogx(lambdas, coefficient_norm_inner,'k')
plt.ylabel('L2 Norm')
plt.xlabel('Regularization strength, $\log_{10}(\lambda)$')
plt.title('Parameter vector L2 norm')
plt.grid()
plt.show()    

# =============================================================================
# 
# #       Train the model with specific lambda = 100 
# 
# =============================================================================

k = 0 

for train_index_outer, test_index_outer in skf_outer.split(X, y):
    print('\nOuter fold: ',k, '/', K, '\n')
    # Extract training and test set for current CV fold
    X_train_outer = X[train_index_outer]
    y_train_outer = y[train_index_outer]
    X_test_outer = X[test_index_outer]
    y_test_outer = y[test_index_outer]



    # Build the model for the optimal lambda
    # Train Optimal model with lambda calculated in the inner loop and compute the generalised error
    mdl_outer = LogisticRegression(multi_class='multinomial',
                                      penalty='l2', random_state=(1),
                                      C=1/100, 
                                      max_iter= 5000)

    # Fit the model 
    mdl_outer.fit(X_train_outer,y_train_outer)
    # Estimate the classes 
    y_train_est_outer = mdl_outer.predict(X_train_outer)
    y_test_est_outer = mdl_outer.predict(X_test_outer)
    # Store the predictions 
    prediction_outer.append(y_test_est_outer)
    y_test_outer_store.append(y_test_outer)
    # Estimate the error of the model for each lambda and inner fold 
    train_error_rate_outer[k] = np.sum(y_train_est_outer!=y_train_outer) / len(y_train_outer)
    test_error_rate_outer[k] = np.sum(y_test_est_outer!=y_test_outer) / len(y_test_outer)    
    # Get attributes
    w_est_outer[k,:] = mdl_outer.coef_[0] 
    coefficient_norm_outer[k] = np.sqrt(np.sum(w_est_outer**2))
    k +=1 





