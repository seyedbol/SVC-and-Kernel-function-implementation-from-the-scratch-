% File: predictUsingSVM.m
% ------------------------------------------------------------
% This function will take the values 'w' and 'b' of a trained SVM model and
% it will use them to predict the class of every sample in the matrix
% dataSamples.
%
% Inputs:
%   - dataSamples: Matrix of 'numSamples' by 'numFeatures' containig the
%     feature vector of the samples whose class will be predicted.
%   - w: weight vector of length 'numFeatures' for the case of the primal,
%     and length 'numSupportVectors' for the case of the dual.
%   - b: bias term (Scalar).
%   - sv: Matrix of numSupportVectors x numFeatures that contains the 
%     support vectors. This value should be empty for the case of 
%     the primal.
%   - sv_label: Vector of numSupportVectors x1 that contains the label of the
%       support vectors. This value should be empty for the case of 
%       the primal.
%   - kernel: String that specifies the kernel to use. It can take the
%     following values:
%       + lin_primal:   for the SVM using the primal.
%       + lin_dual:     for the linear kernel using the dual.
%       + rbf:          for using the RBF kernel
%       + poly:         for using the polynomial kernel
% - params: It is a structure that contains the parameters needed for
%   different versions of the SVM. It contains three values:
%       + params.C:     Soft margin parameter C. (scalar)
%       + params.Sigma: Width of the Gaussian kernel. (scalar)
%       + params.D:     Degree of the polynomial. (discrete)
%
% Outputs:
%   - predictions: column vector of length 'numSamples' that contains a
%     value of -1 or 1 indicating the predicted class of every sample.
function predictions = predictUsingSVM(dataSamples,w,b,sv,sv_labels,...
    kernel,params)

 m= size(dataSamples,1) ;
 k= size(sv,1) ;
%------------------------
if(kernel=="lin_primal")%%%%%+lin_primal:for the SVM using the primal.

for i=1:1:m
   predictions(i,1)= sign( dataSamples(i,:)*w+b )  ;
end
   
end
%----------------------------------------
if(kernel=="lin_dual")%%%%%+lin_dual:for the linear kernel using the dual.  

for i=1:1:m
    a=zeros(k,1) ;
    for j=1:1:k
a(j)= polynomialkernel(dataSamples(i,:),sv(j,:),1) ;    
    end
predictions(i,1)=sign(sum(w.*sv_labels.*a)+b) ;
end  

end

%%%%%

 
 if(kernel=="rbf") 

for i=1:1:m
a=[] ;
    for j=1:1:k
a= [a;gaussiankernel(dataSamples(i,:),sv(j,:),params.Sigma)] ;   

    end
predictions(i,1)=sign(sum(w.*sv_labels.*a)+b) ;
end  

 end
 
 
 
 %%%%%%%%%%%%%%%%%%
 if(kernel=="poly")%%%%%%+poly:for using the polynomial kernel        

for i=1:1:m
    a=zeros(k,1) ;
    for j=1:1:k
a(j)= polynomialkernel(dataSamples(i,:),sv(j,:),params.D) ;    
    end
predictions(i,1)=sign(sum(w.*sv_labels.*a)+b) ;
end  
end



end