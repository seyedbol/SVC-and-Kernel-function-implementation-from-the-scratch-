% This function will estimate the weights vector and the bias term needed
% for making predictions with a support vector machine.
%
% Inputs:
% - dataSamples: matrix of 'numSamples' x 'numFeatures' containing the data
%   that will be used for training the SVM.
% - dataLabels: vector of length 'numSamples' that contains the class of
%   every sample in the matrix 'dataSamples'.
% - kernel: String that specifies the kernel to use. It can take the
%   following values:
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
% - alpha: Vector of length numSamples that contains the value of the
%       Lagrange multipliers (you will use them for the graphs). It can be
%       empty for the primal.
% - w:  Weight vector of length 'numFeatures' for the case of the primal.
%       Alpha vector of length 'numSupportVectors' for the case of the dual.
% - b:  Bias term (scalar).
% - sv: Matrix of numSupportVectors x numFeatures that contains the 
%       support vectors. This value should be empty for the case of 
%       the primal.
% - sv_labels: Vector of numSupportVectors x1 that contains the label of the
%       support vectors. This value should be empty for the case of 
%       the primal. 
%
% ----------------------------------------------------------------------
%             Note about the Support Vectors (for the dual)
% ----------------------------------------------------------------------
%       Remember that the support vectors are those whose value
%       of alpha is > 0 (In practice, you will not get a value of exactly
%       zero, so consider as support vectors those whos value of alpha is >
%       .0001).
%
%       For computing the vale of the bias term 'b', you can take the
%       average over the values of 'b' obtained using the alpha value and
%       label of every support vector.
% ----------------------------------------------------------------------

function [alpha,w,b,sv,sv_labels] = trainSVM_model(dataSamples, dataLabels, kernel, params)
%--------------------- 
 m=size(dataLabels,1) ;
 n=size(dataSamples,2);
%----------------------
 if(kernel=="lin_primal")%primal without kernel
   
   %parameter c
   H=diag([ones(1,n) zeros(1,m+1)]) ;
   b=-1*ones(m,1);
   f=params.C*[zeros(n+1,1);ones(m,1)];
   A=-diag(dataLabels)*[dataSamples ones(m,1) zeros(m,m)] -[zeros(m,n+1) eye(m)] ;
   lb=[-inf.*ones(n+1,1);zeros(m,1)];
   ub=inf.*ones(m+n+1,1);
   z=quadprog(H,f,A,b,[],[],lb,ub);
   w=z(1:n,1);
   b=z(n+1,1);
   sv=[];
   sv_labels=[];
   alpha=[];     
     
 end
 %----------------------
 if(kernel=="lin_dual")%%dual from with linear kernel
 
  % parameter C
  H=zeros(m,m);
  for i=1:m
      for j=1:m
  H(i,j)= dataLabels(i)*dataLabels(j)*polynomialkernel(dataSamples(i,:),dataSamples(j,:),1);
      end
  end
  f=-params.C.*ones(m,1);
  A=[];
  b=[];        
  Beq=0;
  Aeq=dataLabels';
  lb=zeros(m,1);
  ub=ones(m,1);
  alpha=quadprog(H,f,A,b,Aeq,Beq,lb,ub);
  alphaindex=find(alpha>0.0001) ;
  w=alpha(alphaindex) ;
  sv=dataSamples(alphaindex,:) ;
  sv_labels=dataLabels(alphaindex) ;
  %%%calculating b(go to each support vector  and get the kernel vector for
  %%%all the datasamples .then, multiply it by alpha and datlabels and
  %%sum it over all thes supportvectors (very tricky!!!:)))
  
   for i=1:1:size(sv_labels,1)
       A=0 ;
       for j=1:1:size(dataLabels,1)
           A=A+polynomialkernel(sv(i,:),dataSamples(j,:),1)*alpha(j)*dataLabels(j)  ; %%this function calculate the kernel for polynomials 
       end
       bvector(i)=sv_labels(i)-A  ;
   end
  b = sum(bvector)/size(sv_labels,1) ;%%%take the average over support vectors
  
 end
 %-----------------------
 if(kernel=="rbf")%%gaussian kernel
  
  % parameter C&Sigma
  H=zeros(m,m);
  for i=1:m
      for j=1:m
  H(i,j)= dataLabels(i)*dataLabels(j)*gaussiankernel(dataSamples(i,:),dataSamples(j,:),params.Sigma);
      end
  end
  f=-params.C.*ones(m,1);
  A=[];
  b=[];        
  Beq=0;
  Aeq=dataLabels';
  lb=zeros(m,1);
  ub=ones(m,1);
  alpha=quadprog(H,f,A,b,Aeq,Beq,lb,ub);
  alphaindex=find(alpha>0.0001) ;
  w=alpha(alphaindex) ;
  sv=dataSamples(alphaindex,:) ;
  sv_labels=dataLabels(alphaindex) ;
  %%%calculating b(go to each support vector  and get the kernel vector for
  %%%all the datasamples .then, multiply it by alpha and datlabels and
  %%sum it over all thes supportvectors (very tricky!!!:)))
  
   for i=1:1:size(sv_labels,1)
       A=0 ;
       for j=1:1:size(dataLabels,1)
           A=A+gaussiankernel(sv(i,:),dataSamples(j,:),params.Sigma)*alpha(j)*dataLabels(j)  ; %%this function calculate the kernel for polynomials 
       end
       bvector(i)=sv_labels(i)-A  ;
   end
  b = sum(bvector)/size(sv_labels,1) ;%%%take the average over support vectors
 end
 %-----------------------
 if(kernel=="poly")%%polynomial kernel

  % parameter C&D
  H=zeros(m,m);
  for i=1:m
      for j=1:m
  H(i,j)= dataLabels(i)*dataLabels(j)*polynomialkernel(dataSamples(i,:),dataSamples(j,:),params.D);
      end
  end
  f=-params.C.*ones(m,1);
  A=[];
  b=[];        
  Beq=0;
  Aeq=dataLabels';
  lb=zeros(m,1);
  ub=ones(m,1);
  alpha=quadprog(H,f,A,b,Aeq,Beq,lb,ub);
  alphaindex=find(alpha>0.0001) ;
  w=alpha(alphaindex) ;
  sv=dataSamples(alphaindex,:) ;
  sv_labels=dataLabels(alphaindex) ;
  %%%calculating b(go to each support vector  and get the kernel vector for
  %%%all the datasamples .then, multiply it by alpha and datlabels and
  %%sum it over all thes supportvectors (very tricky!!!:)))
  
   for i=1:1:size(sv_labels,1)
       A=0 ;
       for j=1:1:size(dataLabels,1)
           A=A+polynomialkernel(sv(i,:),dataSamples(j,:),params.D)*alpha(j)*dataLabels(j)  ; %%this function calculate the kernel for polynomials 
       end
       bvector(i)=sv_labels(i)-A  ;
   end
  b = sum(bvector)/size(sv_labels,1) ;%%%take the average over support vectors
end