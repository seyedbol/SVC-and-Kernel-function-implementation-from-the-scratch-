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

function generateModelsToBeGradedtest()

global p1 ;
% % For the Survival Dataset
% % Write here your code for generating 'w', 'b', sv, and sv_labels  for the
% % different scenarios
% 
Name of the variables:
w_lin_primal = 0;   b_lin_primal = 0;
w_lin_dual = 0;     b_lin_dual = 0;     sv_lin = 0;     sv_labels_lin = 0;
w_rbf = 0;          b_rbf = 0;          sv_rbf = 0;     sv_labels_rbf = 0;
w_poly = 0;         b_poly = 0;         sv_poly = 0;    sv_labels_poly = 0;
bestParam_lin_primal = 0; bestParam_lin_dual = 0; bestParam_rbf = 0;
bestParam_poly = 0;
% % % ------------------------------------------------
% % % ------------------------------------------------
load('iris1_v24.mat');
load('Folds_Iris.mat');
% % 
%   kernel='lin_primal';
%   bestParam_lin_primal = findBestParams(X_train, Y_train, external_fold, kernel);
%   [alpha,w_lin_primal,b_lin_primal,sv_lin,sv_labels_lin] = trainSVM_model(X_train, Y_train,kernel, bestParam_lin_primal);
%   acc_lin_primal = expectedAccuracy(X_train, Y_train, 'lin_primal',external_fold, internal_fold);
% 
%   title('Survival lin_primal classification');
%   figure(1)
%   Xn= X_train(find(Y_train==-1),:);
%   Xp= X_train(find(Y_train==1),:); 
%   plot(Xn(:,1),Xn(:,2),'go');
%   hold on
%   plot(Xp(:,1),Xp(:,2),'r*');
%   hold on
%         
%   plot_x = [min(X_train(:,1))-2,  max(X_train(:,1))+2];
%       % Calculate the decision boundary line
%   plot_y = (-1./w_lin_primal(2)).*(w_lin_primal(1).*plot_x + b_lin_primal);
%       %Plot, and adjust axes for better viewing
%   plot(plot_x, plot_y)
% 
% 
% kernel='lin_dual';
%   bestParam_lin_dual = findBestParams(X_train, Y_train, external_fold, kernel);
%   [alpha,w_lin_dual,b_lin_dual,sv_lin_dual,sv_labels_lin_dual] = trainSVM_model(X_train, Y_train,kernel, bestParam_lin_dual) ;
%   acc_lin_dual = expectedAccuracy(X_train, Y_train, kernel, external_fold, internal_fold);
%   figure(2)
%   title('Survival lin_dual classification');
%   svcplot(X_train,Y_train,kernel,alpha,b_lin_dual);
%   disp('b is')
%   disp(b_lin_dual)
%   disp('C is man')
%   disp(bestParam_lin_dual.C)
      
% 
% 
kernel='rbf'
%   bestParam_rbf = findBestParams(X_train, Y_train, external_fold, 'rbf');
%   [alpha,w_rbf,b_rbf,sv_rbf,sv_labels_rbf] = trainSVM_model(X_train, Y_train,'rbf', bestParam_rbf) ;
acc_rbf = expectedAccuracy(X_train, Y_train, 'rbf', external_fold, internal_fold);
  p1= bestParam_rbf.Sigma;
  disp("p1 is")
  disp(p1)
  disp('b is')
  disp(b_rbf)
  disp('C is man')
  disp(bestParam_rbf.C)
  title('Survival rbf classification');
  svcplot(X_train,Y_train,kernel,alpha,b_rbf);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% it is ok
% kernel='poly'
% %     bestParam_poly = findBestParams(X_train, Y_train, external_fold, kernel);
% %     [alpha,w_poly,b_poly,sv_poly,sv_labels_poly] = trainSVM_model(X_train, Y_train,kernel, bestParam_poly) ;
% %     acc_poly= expectedAccuracy(X_train, Y_train, kernel, external_fold, internal_fold);
% %     p1=bestParam_poly.D;
% %     figure(4)
% %     svcplot(X_train,Y_train,kernel,alpha,b_poly);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



clear all;
