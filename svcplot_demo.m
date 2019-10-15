% You need to set the value of this global variable in your code. Check the
% function svkernel.m to find which are the values that 'p1' has to take.
global p1
p1=.4;

load iris1_v24;
alpha=randn(size(X,1),1);
% Check the comments in the function svcplot to learn how to use it.
svcplot(X,Y,'rbf',alpha,0);
