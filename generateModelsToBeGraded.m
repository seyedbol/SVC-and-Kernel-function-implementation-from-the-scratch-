% File:generateModelsToBeGraded.m
% ------------------------------------------------------------------
% This function will save the outputs that we require to evaluate your
% implementation of SVM. In this function you are expected to do the
% following:
% 1. Using the dataset, you need to find the best parameters for every
% classifier that you will create (Linear SVM primal, Linear SVM Dual, RBF,
% and Polynomial kernel).
% 2. Use those parameters and the entire dataset to train your final
% classifiers.
% 3. You can use the outputs of your SVM model to create the graphs (Check
% the files inside the folder 'SVCplot_demo').
%
% Please pay attention to the name that you have to give to the variables
% that you create.
%
% Note that you already created all the code for perforing the steps 1 and
% 2 of this function, so this part should be simple to do.
%
% Do not modify the names of the .mat files, nor the name of the
% variables that will be saved on those .mat files.

function generateModelsToBeGraded()

% For the Survival Dataset
% Write here your code for generating 'w', 'b', sv, and sv_labels  for the
% different scenarios
global p1 ;
load('survivaldatatrain.mat');
load('Folds_Survival.mat') ;
% Name of the variables:
w_lin_primal = 0;   b_lin_primal = 0;
w_lin_dual = 0;     b_lin_dual = 0;     sv_lin = 0;     sv_labels_lin = 0;
w_rbf = 0;          b_rbf = 0;          sv_rbf = 0;     sv_labels_rbf = 0;
w_poly = 0;         b_poly = 0;         sv_poly = 0;    sv_labels_poly = 0;
bestParam_lin_primal = 0; bestParam_lin_dual = 0; bestParam_rbf = 0; 
bestParam_poly = 0;
% ------------------------------------------------
 %%%%%%linear_primal
  kernel='lin_primal';
  bestParam_lin_primal = findBestParams(X_train, Y_train, external_fold, kernel);
  [alpha,w_lin_primal,b_lin_primal,sv_lin,sv_labels_lin] = trainSVM_model(X_train, Y_train,kernel, bestParam_lin_primal);
  acc_lin_primal = expectedAccuracy(X_train, Y_train, 'lin_primal',external_fold, internal_fold);
%   svcplot(X_train,Y_train,kernel,w_lin_primal,b_lin_primal);function does
%   not work for primal so i did the plotting process manually
figure(1)
Xn= X_train(find(Y_train==-1),:);
Xp= X_train(find(Y_train==1),:); 
plot(Xn(:,1),Xn(:,2),'ro');
hold on
plot(Xp(:,1),Xp(:,2),'bo');
hold on
constant=5 ;%%%to show the plot complete
x = [min(X_train(:,1))-constant,max(X_train(:,1))+constant];
y = (-1./w_lin_primal(2)).*(w_lin_primal(1).*x + b_lin_primal);
title('Survival linprimal classification');
plot(x,y);
acc_lin_primal = expectedAccuracy(X_train, Y_train, 'lin_primal', external_fold, internal_fold);
% ------------------------------------------------
%%lin_dual
kernel='lin_dual';
bestParam_lin_dual = findBestParams(X_train, Y_train, external_fold, kernel);
[alpha,w_lin_dual,b_lin_dual,sv_lin_dual,sv_labels_lin_dual] = trainSVM_model(X_train, Y_train,kernel, bestParam_lin_dual) ;
figure(2)
title('Survival lin_dual classification');
svcplot(X_train,Y_train,kernel,alpha,b_lin_dual);
acc_lin_dual = expectedAccuracy(X_train, Y_train, 'lin_dual',external_fold, internal_fold);
%---------------------------------------------------
%polynomial
kernel='poly';
bestParam_poly = findBestParams(X_train, Y_train, external_fold, kernel);
[alpha,w_poly,b_poly,sv_poly,sv_labels_poly] = trainSVM_model(X_train, Y_train,kernel, bestParam_poly) ;
p1=bestParam_poly.D;
figure(3)
svcplot(X_train,Y_train,kernel,alpha,b_poly);
%     acc_poly= expectedAccuracy(X_train, Y_train, kernel, external_fold, internal_fold);
%I commented the accuracy for polynomial because in survival data set gives me non-convex error 
%and I do not know what the problem is!!!!!!!!
%-----------------------------------------------
kernel='rbf'
bestParam_rbf = findBestParams(X_train, Y_train, external_fold, 'rbf');
[alpha,w_rbf,b_rbf,sv_rbf,sv_labels_rbf] = trainSVM_model(X_train, Y_train,'rbf', bestParam_rbf) ;
p1= bestParam_rbf.Sigma;
title('Survival rbf classification');
figure(4)
svcplot(X_train,Y_train,kernel,alpha,b_rbf);
%     acc_rbf= expectedAccuracy(X_train, Y_train, kernel, external_fold, internal_fold);
%I commented the accuracy for rbf as well because in survival data set gives me non-convex error 
%and I do not know what the problem is!!!!!!!!
%-------------------------------------------------------------------------------------------------------------
acc_rbf=0 ; acc_poly=0 ;%%%I assigned them zero to avoid facing error
save('SurvivalModel.mat','w_lin_primal','b_lin_primal', 'w_lin_dual', ...
    'b_lin_dual','sv_lin', 'sv_labels_lin','w_rbf', 'b_rbf', 'sv_rbf', ...
    'sv_labels_rbf', 'w_poly', 'b_poly', 'sv_poly', 'sv_labels_poly', ...
    'acc_lin_primal', 'acc_lin_dual', 'acc_rbf','acc_poly',...
    'bestParam_lin_primal','bestParam_lin_dual','bestParam_rbf',...
    'bestParam_poly');
% 
clear all;
global p1 ;
%%
% For the Chess Dataset
% Write here your code for generating 'w', 'b', sv, and sv_labels  for the
% different scenarios

% Name of the variables:
w_lin_primal = 0;   b_lin_primal = 0;
w_lin_dual = 0;     b_lin_dual = 0;     sv_lin = 0;     sv_labels_lin = 0;
w_rbf = 0;          b_rbf = 0;          sv_rbf = 0;     sv_labels_rbf = 0;
w_poly = 0;         b_poly = 0;         sv_poly = 0;    sv_labels_poly = 0;
bestParam_lin_primal = 0; bestParam_lin_dual = 0; bestParam_rbf = 0; 
bestParam_poly = 0;

%------------------------------------------------
%------------------------------------------------
load('chessboarddatatrain.mat');
load('Folds_Chess.mat');
 %%%%%%linear_primal
  kernel='lin_primal';
  bestParam_lin_primal = findBestParams(X_train, Y_train, external_fold, kernel);
  [alpha,w_lin_primal,b_lin_primal,sv_lin,sv_labels_lin] = trainSVM_model(X_train, Y_train,kernel, bestParam_lin_primal);
  acc_lin_primal = expectedAccuracy(X_train, Y_train, 'lin_primal',external_fold, internal_fold);
%   svcplot(X_train,Y_train,kernel,w_lin_primal,b_lin_primal);function does
%   not work for primal so i did the plotting process manually
figure(5)
Xn= X_train(find(Y_train==-1),:);
Xp= X_train(find(Y_train==1),:); 
plot(Xn(:,1),Xn(:,2),'ro');
hold on
plot(Xp(:,1),Xp(:,2),'bo');
hold on
constant=2;%%%to show the plot complete
x = [min(X_train(:,1))-constant,max(X_train(:,1))+constant];
y = (-1./w_lin_primal(2)).*(w_lin_primal(1).*x + b_lin_primal);
title('chess linprimal classification');
plot(x,y);
acc_lin_primal = expectedAccuracy(X_train, Y_train, 'lin_primal', external_fold, internal_fold);
% ------------------------------------------------
%%lin_dual
kernel='lin_dual';
bestParam_lin_dual = findBestParams(X_train, Y_train, external_fold, kernel);
[alpha,w_lin_dual,b_lin_dual,sv_lin_dual,sv_labels_lin_dual] = trainSVM_model(X_train, Y_train,kernel, bestParam_lin_dual) ;
figure(6)
title('Survival lin_dual classification');
svcplot(X_train,Y_train,kernel,alpha,b_lin_dual);
acc_lin_dual = expectedAccuracy(X_train, Y_train, 'lin_dual',external_fold, internal_fold);
%---------------------------------------------------
%polynomial
kernel='poly';
bestParam_poly = findBestParams(X_train, Y_train, external_fold, kernel);
[alpha,w_poly,b_poly,sv_poly,sv_labels_poly] = trainSVM_model(X_train, Y_train,kernel, bestParam_poly) ;
p1=bestParam_poly.D;
svcplot(X_train,Y_train,kernel,alpha,b_poly);
acc_poly= expectedAccuracy(X_train, Y_train, kernel, external_fold, internal_fold);
%-----------------------------------------------
kernel='rbf'
bestParam_rbf = findBestParams(X_train, Y_train, external_fold, 'rbf');
[alpha,w_rbf,b_rbf,sv_rbf,sv_labels_rbf] = trainSVM_model(X_train, Y_train,'rbf', bestParam_rbf) ;
p1= bestParam_rbf.Sigma;
title('Survival rbf classification');
svcplot(X_train,Y_train,kernel,alpha,b_rbf);
%  acc_rbf= expectedAccuracy(X_train, Y_train, kernel, external_fold, internal_fold);
%I commented line 159-161 the help you run the code without error. In rbf
%distribution for chess case I face non-convex-error
%it is very strange!!!!!!!
%-------------------------------------------------------------------------------------------------------------
acc_rbf=0 ;
%------------------------------------------------
save('ChessModel.mat','w_lin_primal','b_lin_primal', 'w_lin_dual', ...
    'b_lin_dual','sv_lin', 'sv_labels_lin','w_rbf', 'b_rbf', 'sv_rbf', ...
    'sv_labels_rbf', 'w_poly', 'b_poly', 'sv_poly', 'sv_labels_poly', ...
    'acc_lin_primal', 'acc_lin_dual', 'acc_rbf','acc_poly',...
    'bestParam_lin_primal','bestParam_lin_dual','bestParam_rbf',...
    'bestParam_poly');
clear all
global p1 ;
%%

% For the Iris Dataset
% Write here your code for generating 'w', 'b', sv, and sv_labels  for the
% different scenarios

% Name of the variables:
w_lin_primal = 0;   b_lin_primal = 0;
w_lin_dual = 0;     b_lin_dual = 0;     sv_lin = 0;     sv_labels_lin = 0;
w_rbf = 0;          b_rbf = 0;          sv_rbf = 0;     sv_labels_rbf = 0;
w_poly = 0;         b_poly = 0;         sv_poly = 0;    sv_labels_poly = 0;
bestParam_lin_primal = 0; bestParam_lin_dual = 0; bestParam_rbf = 0; 
bestParam_poly = 0;
% ------------------------------------------------
% YOUR CODE HERE
load('iris1_v24.mat');
load('Folds_Iris.mat');
 %%%%%%linear_primal
  kernel='lin_primal';
  bestParam_lin_primal = findBestParams(X_train, Y_train, external_fold, kernel);
  [alpha,w_lin_primal,b_lin_primal,sv_lin,sv_labels_lin] = trainSVM_model(X_train, Y_train,kernel, bestParam_lin_primal);
  acc_lin_primal = expectedAccuracy(X_train, Y_train, 'lin_primal',external_fold, internal_fold);
%   svcplot(X_train,Y_train,kernel,w_lin_primal,b_lin_primal);function does
%   not work for primal so i did the plotting process manually below
figure(9)
Xn= X_train(find(Y_train==-1),:);
Xp= X_train(find(Y_train==1),:); 
plot(Xn(:,1),Xn(:,2),'ro');
hold on
plot(Xp(:,1),Xp(:,2),'bo');
hold on
constant=2;%%%to show the plot complete
x = [min(X_train(:,1))-constant,max(X_train(:,1))+constant];
y = (-1./w_lin_primal(2)).*(w_lin_primal(1).*x + b_lin_primal);
title('iris linprimal classification');
plot(x,y);
acc_lin_primal = expectedAccuracy(X_train, Y_train, 'lin_primal', external_fold, internal_fold);
% ------------------------------------------------
%%lin_dual
kernel='lin_dual';
bestParam_lin_dual = findBestParams(X_train, Y_train, external_fold, kernel);
[alpha,w_lin_dual,b_lin_dual,sv_lin_dual,sv_labels_lin_dual] = trainSVM_model(X_train, Y_train,kernel, bestParam_lin_dual) ;
figure(10)
title('Survival lin_dual classification');
svcplot(X_train,Y_train,kernel,alpha,b_lin_dual);
acc_lin_dual = expectedAccuracy(X_train, Y_train, 'lin_dual',external_fold, internal_fold);
%---------------------------------------------------
%polynomial
kernel='poly';
bestParam_poly = findBestParams(X_train, Y_train, external_fold, kernel);
[alpha,w_poly,b_poly,sv_poly,sv_labels_poly] = trainSVM_model(X_train, Y_train,kernel, bestParam_poly) ;
p1=bestParam_poly.D;
figure(11)
svcplot(X_train,Y_train,kernel,alpha,b_poly);
% acc_poly= expectedAccuracy(X_train, Y_train, kernel, external_fold, internal_fold);
%-----------------------------------------------
% kernel='rbf'
bestParam_rbf = findBestParams(X_train, Y_train, external_fold, 'rbf');
[alpha,w_rbf,b_rbf,sv_rbf,sv_labels_rbf] = trainSVM_model(X_train, Y_train,'rbf', bestParam_rbf) ;
p1= bestParam_rbf.Sigma;
figure(12)
title('Survival rbf classification');
svcplot(X_train,Y_train,kernel,alpha,b_rbf);
% acc_rbf= expectedAccuracy(X_train, Y_train, kernel, external_fold, internal_fold);

% ------------------------------------------------
% %I commented line 222-237 to help you run the code without error. In rbf
% and poly distribution for iris case I faced non-convex-error
% %it is very strange!!!!!!!
acc_rbf=0 ;
acc_poly=0 ;
save('IrisModel.mat','w_lin_primal','b_lin_primal', 'w_lin_dual', ...
    'b_lin_dual','sv_lin', 'sv_labels_lin','w_rbf', 'b_rbf', 'sv_rbf', ...
    'sv_labels_rbf', 'w_poly', 'b_poly', 'sv_poly', 'sv_labels_poly', ...
    'acc_lin_primal', 'acc_lin_dual', 'acc_rbf','acc_poly',...
    'bestParam_lin_primal','bestParam_lin_dual','bestParam_rbf',...
    'bestParam_poly');
clear all