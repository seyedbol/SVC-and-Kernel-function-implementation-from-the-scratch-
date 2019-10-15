function A = binaryaccuracy(predictions,lables)
k = size(lables,1) ;
accuracy=zeros(k,1) ;
%%%if prediction equals reality this function gives 1
%%%if prediction does not equal reality this function gives 0
for i=1:1:k
    if(predictions(i,1)==lables(i,1))
    accuracy(i,1)= 1 ;   
    end

end
%%take the average over vector
A=sum(accuracy)/k;

end