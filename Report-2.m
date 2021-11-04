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


























