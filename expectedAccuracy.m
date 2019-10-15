% This function will estimate the performance of your learning algorithm on
% new, unseen data using 5-fold cross validation. The general procedure is
% as follows:
%
% For every interation i (i = 1:5 in this case) :
%   - Create a trainSet and testSet (using the vector 'external_folds') 
%   - Using the trainSet and the function findBestParams.m, find optimal
%     values for the SVM parameters. Note that this function internally
%     uses 5-fold CV too, so you need to provide a vector that asigns every
%     sample in the trainSet to a fold. You will use the vector stored at
%     'internal_fold{i}' (Note the use of {}, instead of parenthesis).
%   - Using the bestParameters create a SVM model using the trainSet.
%   - Use the trained model to make predictions on the testSet.
%
% Finally, output the average accuracy over the 5 iterations.

% Inputs:
% - dataSamples: matrix of 'numSamples' x 'numFeatures' containing the data
%   that will be used for training the SVM.
% - dataLabels: vector of length 'numSamples' that contains the class of
%   every sample in the matrix 'dataSamples'.
% - kernel: String that specifies the kernel to use. It can take the
%   following values:
%       + lin_primal:   for the linear kernel using the primal.
%       + lin_dual:     for the linear kernel using the dual.
%       + rbf:          for using the RBF kernel
%       + poly:         for using the polynomial kernel
% - external_folds: vector that assignes every sample in dataSamples to a
%   fold.
% - internal_folds: cell with K = 5 entries that contains the vector for
%   internal cross-validation. The internal folds will be used when finding
%   the optimal parameters for the SVM.
%
% ----------------------------------------------------------------------
%                            IMPORTANT
% ----------------------------------------------------------------------
% The vector folds assign every sample into a fold for performing 5-fold
% cross validation. This means that in every iteration you will have to
% create a new training set a test set. In the first iteration, the entries
% with a 1 in the vector 'external_folds' are part of the TEST set. 
% In the second iteration, the TEST set are the entries with a 2 in the 
% vector 'external_folds' and so on.
%
% When calling the function findBestParams you will need to pass a new
% vector fold. You will need to use internal_fold{i} for the ith iteration.
% ----------------------------------------------------------------------
function accuracy = expectedAccuracy(dataSamples, dataLabels, kernel, ...
    external_folds, internal_folds)
ACC=zeros(1,5);    
for i=1:1:5 %%%external cross validation to get the best accuracy over all the options
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5    
        
        test_data=dataSamples(find(external_folds==i),:);
        test_labels= dataLabels(find(external_folds==i),:);
             
        train_data=dataSamples(find(external_folds~=i),:);
        train_labels= dataLabels(find(external_folds~=i),:);
        
        params = findBestParams(train_data, train_labels, internal_folds{i}, kernel);
        [alpha,w,b,sv,sv_labels] = trainSVM_model(train_data, train_labels,kernel, params);
        predictions = predictUsingSVM(test_data,w,b,sv,sv_labels, kernel,params);
        ACC(i)=binaryaccuracy(predictions,test_labels);  
end

accuracy= mean(ACC);
disp('accuracy is')
disp(accuracy)
end






