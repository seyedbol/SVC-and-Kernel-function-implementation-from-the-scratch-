function[ phi ]= gaussiankernel(u,v,sigma)
A= norm(u-v) ;
phi =exp(-(A.^2/(2*sigma^2) ) ) ;  

end