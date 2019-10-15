function [phi]= polynomialkernel(u,v,D)
global p1
phi = (u*v'+1)^D; %%imported from sv_kernel

end