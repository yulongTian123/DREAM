function [result]= myNMIACCwithmean1(U,Y,numclass)

stream = RandStream.getGlobalStream;
reset(stream);
U_normalized = U ./ repmat(sqrt(sum(U.^2, 2)), 1,size(U,2));
maxIter = 5;
for i = 1:maxIter
indx = kmeans(U_normalized,numclass,'MaxIter',100, 'Replicates',3); %3
%indx = litekmeans(U_normalized,numclass,'MaxIter',100, 'Replicates',3);
indx = indx(:);
result = Clustering8Measure(Y,indx); % result = [ACC nmi Purity Fscore Precision Recall AR Entropy];
%resmax = max(result,[],1);
%resmax = result;
%resmax = mean(result,1);
%resstd = std(result);
%resmax = mean(result,1);

end