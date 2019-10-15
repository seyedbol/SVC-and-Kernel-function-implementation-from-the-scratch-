% File: findBestParams
% ------------------------------------------------------------------
% This function will use 5-fold cross validation to determine the best
% parameters for the SVM.
%
% Input:
% - dataSamples: matrix of 'numSamples' x 'numFeatures' containing the data
%   that will be used for training the SVM.
% - dataLabels: vector of length 'numSamples' that contains the class of
%   every sample in the matrix 'dataSamples'.
% - folds: Contains the fold number of every sample in dataSamples. This
%   vector is meant to be used in the cross validation process. For example,
%   in the ith iteration all the samples whose entry in folds == i will be
%   part of the test set, while the rest will be used for trainig the model.
% - kernel: String that specifies the kernel to use. It can take the
%   following values:
%       + lin_primal:   for the linear kernel using the primal.
%       + lin_dual:     for the linear kernel using the dual.
%       + rbf:          for using the RBF kernel
%       + poly:         for using the polynomial kernel
% Output:
% - bestParams: Structure that contains the following fields:
%       + bestParams.C:     Soft margin parameter C.
%       + bestParams.Sigma: Width of the Gaussian kernel.
%       + bestParams.D:     Degree of the polynomial.
%   Note that not all the kernels require the same parameters, if a
%   parameter is not required it can be left blank.

% ----------------------------------------------------------------------
%                            IMPORTANT
% ----------------------------------------------------------------------
% The vector folds assign every sample into a fold for performing 5-fold
% cross validation. This means that in every iteration you will have to
% create a new training set a test set. In the first iteration, the entries
% with a 1 in the vector 'folds' are part of the TEST set. In the second
% iteration, the TEST set are the entries with a 2 in the vector 'folds'
% and so on.
% ----------------------------------------------------------------------


function bestParams = findBestParams(dataSamples, dataLabels, folds, kernel)
    % You need to try all possible combinations of the following
    % parameters, and choose the combination that produces the highest
    % cross-validation accuracy. In case of ties you need to select the
    % lowest possible value of C (among the ties), and then the lowest
    % value of sigma/degree (among the ties).
    C = [2^-5, 2^-3, 2^-1, 2, 2^3, 2^5];
    Sigma = [2^-9, 2^-7, 2^-5, 2^-3, 2^-1, 2, 2^3];
    Degree = [2, 3, 4, 5, 6, 7, 8];
%-------------------------------
 m = size(dataSamples,1) ;
 Csize= size(C,2) ;
 sigmasize=size(Sigma,2);
 Degreesize=size(Degree,2);
%--------------------
if(kernel=="lin_primal")    
        for i=1:1:Csize
            params.C=C(i);
            for j=1:1:5
                sample_for_test=dataSamples(find(folds==j),:);
                label_for_test= dataLabels(find(folds==j),:);
                sample_for_train=dataSamples(find(folds~=j),:);
                label_for_train= dataLabels(find(folds~=j),:);
                %-------------------------------------
                [alpha,w,b,sv,sv_labels] = trainSVM_model(sample_for_train, label_for_train, 'lin_primal', params);
                predictions = predictUsingSVM(sample_for_test,w,b,sv,sv_labels,'lin_primal',params);
                accuracy(j,i)=binaryaccuracy(predictions,label_for_test);
                %------------------------------------------------------
            end
        end
        [maximum,accuracyindex]=max(mean(accuracy));
        bestParams.C=C(accuracyindex);

end



%%%%%%%%%%%
if(kernel=="lin_dual")%%%%%+lin_dual:for the linear kernel using the dual.  
        for i=1:1:Csize
            params.C=C(i);
            for j=1:1:5
                sample_for_test=dataSamples(find(folds==j),:);
                label_for_test= dataLabels(find(folds==j),:);
                sample_for_train=dataSamples(find(folds~=j),:);
                label_for_train= dataLabels(find(folds~=j),:);
                %-------------------------------------
                [alpha,w,b,sv,sv_labels] = trainSVM_model(sample_for_train, label_for_train, 'lin_dual', params);
                predictions = predictUsingSVM(sample_for_test,w,b,sv,sv_labels,'lin_dual',params);
                accuracy(j,i)=binaryaccuracy(predictions,label_for_test);
                %------------------------------------------------------
            end
        end
        [maximum,accuracyindex]=max(mean(accuracy));
        bestParams.C=C(accuracyindex);

end






%%%%%%%%%
 if(kernel=="rbf")
        for k=1:1:sigmasize
            params.Sigma=Sigma(1,k);
            for i=1:1:Csize
                params.C=C(i);        
                for j=1:1:5
                    test_samples=dataSamples(find(folds==j),:);
                    test_labels= dataLabels(find(folds==j),:);
                    train_samples=dataSamples(find(folds~=j),:);
                    train_labels= dataLabels(find(folds~=j),:);
                    [alpha,w,b,sv,sv_labels] = trainSVM_model(train_samples, train_labels,'rbf', params);
                    if(length(w)==0)
                        accuracy(j,i)=0;
                    else
                        predictions = predictUsingSVM(test_samples,w,b,sv,sv_labels,'rbf',params);
                        accuracy(j,i)=binaryaccuracy(predictions,test_labels);
                    end
                end
            end
             accuracycomplete(k,:)=mean(accuracy);
        end
        [x,y]= find(accuracycomplete==max(max(accuracycomplete)));
        bestParams.C=C(y);
        bestParams.Sigma=Sigma(x);
 end
 
 
 
 
 %%%%%%%%%%%%%%%%%%
 if(kernel=="poly")        
        for k=1:1:Degreesize
            params.D=Degree(1,k);
            for i=1:Csize
                params.C=C(i);        
                for j=1:1:5
                    test_samples=dataSamples(find(folds==j),:);
                    test_labels= dataLabels(find(folds==j),:);
                    train_samples=dataSamples(find(folds~=j),:);
                    train_labels= dataLabels(find(folds~=j),:);
                    [alpha,w,b,sv,sv_labels] = trainSVM_model(train_samples, train_labels,'poly', params);
                    if(length(w)==0)
                        accuracy(j,i)=0;
                    else
                        predictions = predictUsingSVM(test_samples,w,b,sv,sv_labels,'poly',params);
                        accuracy(j,i)=binaryaccuracy(predictions,test_labels);
                    end
                end
            end
             accuracycomplete(k,:)=mean(accuracy);
        end
        [x,y]= find(accuracycomplete==max(max(accuracycomplete)));
        bestParams.C=C(y);
        bestParams.D=Degree(x);
end
    
    
end