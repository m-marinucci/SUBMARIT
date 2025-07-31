function [Rand,AdjRand] = RandIndex4(Clusters1,Clusters2)
%Calculates the Rand index of clustering agreement between two clustering
%configurations)
%Inputs
%Clusters1 - A n*1 cluster assignment containing the cluster number
%Clusters2 - A second cluster assignment with same properties
%Outputs
%Rand - The Rand Index from Rand (1971)
%AdjRand - The adjusted Rand index from Hubert and Arabie (1985)
%-------------------------------------------------------------------------
%Version     Author            Date
%   0.10     Stephen France    08/07/2012


n=size(Clusters1,1); % n is the number of items
N=n*(n-1)/2;
NoClus=max(max([Clusters1,Clusters2]));

%Create a cluster table
CA1=zeros(n,NoClus);
CA2=zeros(n,NoClus);
CA1([1:n]'+n*(Clusters1-1))=1;
CA2([1:n]'+n*(Clusters2-1))=1;

Match=CA1'*CA2;
ColSums=sum(Match,1);
RowSums=sum(Match,2);

P=sum(RowSums.*(RowSums-1)/2);
Q=sum(ColSums.*(ColSums-1)/2);
T=(sum(sum(Match.^2))-n)/2;
N=n*(n-1)/2;
Rand=(N+2*T-P-Q)./N;

AdjRand=2*(N*T-P*Q)./(N*(P+Q)-2*P*Q);


