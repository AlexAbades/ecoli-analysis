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
ecoli_att = table2array(X(:, {'mgc', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2'}));

% Append the y encoded vector to the matrix. It's not really essential
ecoli = [ecoli_att, y];
size(ecoli)
% Create a categoriacal vector with the attributes from the data. So we can
% use them as labels later (not sure if it's necessesary)
cat = categorical({'cp', 'im', 'pp', 'imU', 'om', 'omL', 'imL', 'imS'});

%% Count number of instances 
count = accumarray(ic,1);
values = [unique(y), count];

% We could also work with tables, I am not really used to that
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
% set legend 
b = ['cp :143'; 'im : 77'; 'imL: 2 '; 'imS: 2 '; 'imU: 35'; 'om : 20'; 'omL: 5 '; 'pp : 52'];

% Lip Attribute
figure()
subplot(1,2,1)
gscatter(index, ecoli(:,3), classLabels, c_map)
xlabel('Data points')
ylabel('lip Attribute')
ylim([0.40,1.1])
legend(b)

% Chg Attribute
subplot(1,2,2)
gscatter(index, ecoli(:,4), classLabels, c_map)
xlabel('Data points')
ylabel('chg Attribute')
ylim([0.40,1.1])
legend(b)

%{
It would be nice to explain which data point are outside and if we could
consider them as ouliers or if it has something to do with the
categorization. 
Lip 
Cgh Only one data point with value one from two samples 

%}


%% BOX plots

% Box plot to analyse the importance of the mcg variable
figure()
boxplot(ecoli(:,1), y, 'labels', classNames)
xlabel('Categories')
ylabel('Score of mgc')
title('Quantity of mgc in each category')
% If we only focused on the mean, could diferenciate two different
% categories, one that's above 0.5 and and another that's below 0.5.

% We could also say that will have to focus on both goups to see how we
% should separate them.
% It will be nice to see with the different cateories the boxplots. Also to
% color them with the different color categories. If we are consistent with
% the colors it will be easier to understand later the data


%% We can plot something like in page 122 of the book



%% Basic summary  Covariance matrix 

Y = basicStats(ecoli_att);



%% Correlation matrix


ecoli_norm = featureNormalization(ecoli_att);
corr_ecoli = corr(ecoli_norm);
figure()
h = heatmap(corr_ecoli); 
h.XDisplayLabels= {'mgc', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2'};
h.YDisplayLabels = {'mgc', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2'};



% put legend on the heatmap, columns and rows to understand more data
% of those variables that have more correlation, plot one variable against
% the other and categorize the data. 

% Vars 3 and 4 have little correlation with the others because they are
% binary attributes (or vars), it doesn't have much sense to put them in
% the corr matrix
% example:

%% More Plots
att1 = att(1,2:end);

figure()
[~,ax]=plotmatrix(ecoli_att); 
set(findall(gcf,'-property','FontSize'),'FontSize',12)
ax(1,1).YLabel.String='mgc'; 
ax(2,1).YLabel.String='gvh'; 
ax(3,1).YLabel.String='lip'; 
ax(4,1).YLabel.String='chg'; 
ax(5,1).YLabel.String='aac'; 
ax(6,1).YLabel.String='alm1'; 
ax(7,1).YLabel.String='alm2'; 

ax(7,1).XLabel.String='mgc'; 
ax(7,2).XLabel.String='gvh'; 
ax(7,3).XLabel.String='lip';
ax(7,4).XLabel.String='chg'; 
ax(7,5).XLabel.String='aac'; 
ax(7,6).XLabel.String='alm1';
ax(7,7).XLabel.String='alm2'; 

%% PCA 
% Compute PCA with singular value decomposition, specify the economy-size
% funtion to fast computation

% Substract the mean from the data
N_ecoli = ecoli_att - mean(ecoli_att);

[U, S, V] = svd(N_ecoli, 'econ');
% S--> Variance
% V--> Eigenvalues
% U--> Rotation matrix

% Project the data into the pricipal compoenents 
Z = U*S;

% eigenvalues 
E_values = diag(S).^2;

% Variance explianed by the principal components 
rho = diag(S).^2./sum(diag(S).^2);
cum_rho = cumsum(rho);

% Select PCA
k = 7;

%% Scree and Variance Explained

% Specify the treshold 
threshold = 0.90;

% Plot variance explained for all PCA
figure()
subplot(1,2,1)
plot(E_values, 'o-')
grid minor
xlabel('Principal component');
ylabel('Eigenvalue');
title('Scree Plot');
legend('Eigenvalues')

subplot(1,2,2)
hold on
plot(rho, 'x-'); % We plot the varience 
plot(cumsum(rho), 'o-');
plot([0,length(rho)], [threshold, threshold], 'k--');
legend({'Individual','Cumulative','Threshold'}, ...
        'Location','best');
ylim([0, 1]);
xlim([1, length(rho)]);
grid minor
xlabel('Principal component');
ylabel('Variance explained value');
title('Variance explained by principal components');

%% Plot PCA of data 1 Pca axis vs Pca axis 2D



[~,y] = ismember(classLabels, classNames);
uniqueCat = unique(y);
%
% RGB values of your favorite colors: 
figure()
subplot(2,2,1)
% Initialize some axes
view(3)
grid on
hold on
% Plot each group individually: 

for k = 1:length(uniqueCat)
      % Get indices of this particular unique group:
      ind = y == uniqueCat(k);
      % Plot only this group: 
      plot3(Z(ind,1),Z(ind,2),Z(ind,3),'.','color',c_map(k,:),'markersize',12); 
end
xlabel('PCA 1')
ylabel('PCA 2')
zlabel('PCA 3')
title('3 PCA plot')
legend(b)
hold off

subplot(2,2,2)
gscatter(Z(:,1), Z(:,2), classLabels, c_map)
xlabel('PCA 1')
ylabel('PCA 2')
sgtitle('PCA of data') 


subplot(2,2,3)
gscatter(Z(:,1), Z(:,3), classLabels, c_map)
xlabel('PCA 1')
ylabel('PCA 3')
sgtitle('PCA of data') 


subplot(2,2,4)
gscatter(Z(:,2), Z(:,3), classLabels, c_map)
xlabel('PCA 2')
ylabel('PCA 3')
sgtitle('PCA of data') 
legend(b)

%%  Plot the data and the scores of the attributes on the PCA

c = {'mgc', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2'};

% circle radius 1
r = 1;
x = 0;
y = 0;
th = 0:pi/50:2*pi;
xunit = r * cos(th) + x;
yunit = r * sin(th) + y;


figure()
subplot(2,2,2)
axis square
grid on
biplot(V(:,1:2),'scores',Z(:,1:2),'VarLabels',c);
hold on 
plot(xunit, yunit);
pbaspect([1 1 1]);
title('Attribute scores in PCA 1 & PCA 2');
xlabel('Component 1');
ylabel('Component 2');
hold off

subplot(2,2,3)
axis square
grid on
biplot(V(:,2:3),'scores',Z(:,2:3),'VarLabels',c);
hold on 
plot(xunit, yunit);
pbaspect([1 1 1]);
title('Attribute scores in PCA 2 & PCA 3');
xlabel('Component 2');
ylabel('Component 3');
hold off

subplot(2,2,4)
axis square
grid on
biplot(V(:,[1,3]),'scores',Z(:,[1,3]),'VarLabels',c);
hold on 
plot(xunit, yunit);
pbaspect([1 1 1]);
title('Attribute scores in PCA 1 & PCA 3');
xlabel('Component 1');
ylabel('Component 3');
hold off

subplot(2,2,1)
axis square
grid on
biplot(V(:,1:3),'scores',Z(:,1:3),'VarLabels',c);
pbaspect([1 1 1]);
title('Attribute scores in PCA 1 & PCA 2 & PCA 3');
sgtitle('Compoenent Pattern plots');
hold off


%% Create a 3D Plot with the first 3 PCA
[~,y] = ismember(classLabels, classNames);
uniqueCat = unique(y);
%
% RGB values of your favorite colors: 
figure()
% Initialize some axes
view(3)
grid on
hold on
% Plot each group individually: 

for k = 1:length(uniqueCat)
      % Get indices of this particular unique group:
      ind = y == uniqueCat(k);
      % Plot only this group: 
      plot3(Z(ind,1),Z(ind,2),Z(ind,3),'.','color',c_map(k,:),'markersize',20); 
end
xlabel('PCA 1')
ylabel('PCA 2')
zlabel('PCA 3')
title('3 PCA plot')
legend(b)
hold off

%% Save values 

save('Ecoli_values.mat');







