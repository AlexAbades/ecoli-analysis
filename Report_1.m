clc; clear all;
%% Read the data.

X = readtable('ecoli.csv');

% Select attribute names
att = {'prot_name', 'mgc', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'cat'};
X.Properties.VariableNames = att;

X(:,'prot_name');
% If we want to use more than one var use brakets

%{
  1.  Sequence Name: Accession number for the SWISS-PROT database
  2.  mcg: McGeoch's method for signal sequence recognition.
  3.  gvh: von Heijne's method for signal sequence recognition.
  4.  lip: von Heijne's Signal Peptidase II consensus sequence score.
           Binary attribute.
  5.  chg: Presence of charge on N-terminus of predicted lipoproteins.
	   Binary attribute.
  6.  aac: score of discriminant analysis of the amino acid content of
	   outer membrane and periplasmic proteins.
  7. alm1: score of the ALOM membrane spanning region prediction program.
  8. alm2: score of ALOM program after excluding putative cleavable signal
	   regions from the sequence.
%}


%% Label encoding
% Substract the column class from the matrix. The column we wish to predict
classLabels = table2cell(X(:,9));
% Fiter the uniques values
[classNames, ia, ic] = unique(classLabels);
% Create an encode vector of the different categories and define colors for
% the categories.
%{
Transform the different categories to numbers 
  cp  (cytoplasm)                                      0 - red
  im  (inner membrane without signal sequence)         1 - blue               
  pp  (perisplasm)                                     2 - green
  imU (inner membrane, uncleavable signal sequence)    3 - yellow
  om  (outer membrane)                                 4 - magenta
  omL (outer membrane lipoprotein)                     5 - brown = [165,42,42]/255;
  imL (inner membrane lipoprotein)                     6 - orange
  imS (inner membrane, cleavable signal sequence)      7 - pink = [255,105,180]/255;
%}
[~,y] = ismember(classLabels, classNames);
% Substract 1 to the vector so it's starts from 0.
y = y-1;
y_len = length(y);
% Lastly, we determine the number of attributes M, the number of
% observations N and the number of classes C:
[M, N] = size(X);
C = length(classNames);

% Create a matrix with the variables that are really useful. So we can
% create an easy further visual analysis.
ecoli_temp = table2array(X(:, {'mgc', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2'}));
% Append the y encoded vector to the matrix. It's not really essential
ecoli = [ecoli_temp, y];
size(ecoli)
% Create a categoriacal vector with the attributes from the data. So we can
% use them as labels later (not sure if it's necessesary)
cat = categorical({'cp', 'im', 'pp', 'imU', 'om', 'omL', 'imL', 'imS'});

%% Count number of instances 
count = accumarray(ic,1);
class count(1)
f = cell2mat(classNames)
count_cat = [classNames, count];
class count


%% Color map 
red = [1 0 0];
blue = [0 0 1];
green = [0 1 0];
yellow = [1 1 0];
magenta = [1 0 1];
brown = [165,42,42]/255;
orange = [255, 165, 0]/255;
pink = [255,105,180]/255;

c_map = [red;blue;green;yellow;magenta;brown;orange;pink];



%% Visualize Continuous Data 
%{
% Fast analysis:
lib and chg are binaries attributes, we can map them into two values 0 and
1. I think it won't affect the analysis.
Check if we can get some information with histograms. 
As lib and chg are binary attributes we plot them separate. 
%}

% Create some histograms to see how the different variables are spred.
% Set color and number of bins 
c = [0 0.5 0.5];
b = 90;

figure()
subplot(3,2,1)
histogram(ecoli(:,1), b, 'FaceColor', c);
title('mcg')

subplot(3,2,2)
histogram(ecoli(:,2), b, 'FaceColor', c);
title('gvh')

subplot(3,2,3)
histogram(ecoli(:,6), b, 'FaceColor', c);
title('alm1')

subplot(3,2,4)
histogram(ecoli(:,7), b, 'FaceColor', c);
title('alm2')

subplot(3,2,5:6)
histogram(ecoli(:,5), b, 'FaceColor', c);
title('aac')
sgtitle('Continuous variables');

%{
We can clearly see that alm1 and alm2 follow a bimodel distribution. The
peaks of both distributions lay arround:
Alm1 
    peak 1: 0.36
    peak 2: 0.77
Alm2
    peak 1: 0.39
    peak 2: 0.78

In addition we can see different bins height, it can be that some of the
categories tend to have a higer value for Alm1 and Alm2 than others.

Mcg also foloows a bimodal distribution, but the two pdf overlap more, the
separation line it's not that clear.

Gvh seems to have also a bimodal distribution, but the peak of the second
distribution it's quite small compared to the others.

Finally, Acc seem to follow a uniform distribution. 

We can try to crate a histogram plot painting each category 
%}



%% Visualize binary data
% For the binari classes we can make a first analysis by ploting the
% different outcomes

% Create a vector from one to 336, simulating the index positions. 
index = 1:length(ecoli(:,3));

% Lip Attribute
figure()
subplot(1,2,1)
gscatter(index, ecoli(:,3), classLabels, c_map)
xlabel('Data points')
ylabel('lip Attribute')
ylim([0.40,1.1])

% Chg Attribute
subplot(1,2,2)
gscatter(index, ecoli(:,4), classLabels, c_map)
xlabel('Data points')
ylabel('chg Attribute')
ylim([0.40,1.1])

%{

Cgh
%}
%%




%%




lib_values = unique(ecoli(:,3));


%%

% Box plot to analyse the importance of the first variable
figure()
boxplot(ecoli(:,1), y, 'labels', classNames)
xlabel('Categories')
ylabel('Score of mgc')
title('Quantity of mgc in each category')
% If we only focused on the mean, could diferenciate two different
% categories, one that's above 0.5 and and another that's below 0.5.

% We could also say that will have to focus on both goups to see how we
% should separate them


%We can plot something like in page 122 of the book



%% Feature Normalization

[E_norm, mu, sigma] = featureNormalize(ecoli);

function [X_norm, mu, sigma] = featureNormalize(X)
    % FEATURENORMALIZE, Normalizes the data mean 0 and std 1.
    
    mu = mean(X);
    sigma = std(X);
    X_norm = bxsfun(@minus, X, mu);
    X_norm = bxsfun(@rdivide, X_norm, mu);



end


