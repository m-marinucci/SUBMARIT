function [EmpDist]=RandCreateDist(NoItems,NoClusters,NoPoints,MinItems)
%Creates an empirical distribution for a given number of items and number
%of clusters for the Rand index.
%INPUTS
%NoItems - The number od items
%NoClusters - The number of clusters
%NoPoints - The number of data points in the empirical distribution
%OUTPUTS
%A structure "EmpDisp" containing the following
%NoPoint - The number of points in the empirical distribution
%Rand - A NoPoints*1 distribution of the z value of the Rand index (sorted asc)
%AdjRand - A NoPoints*1 distribution of the sum of (PHat-P) (sorted asc)

EmpDist.Rand=zeros(NoPoints,1);
EmpDist.AdjRand=zeros(NoPoints,1);

if ~exist('MinItems','var')
  MinItems=1;
end

for iPoint=1:NoPoints
  
  MinItemCount=0;   %Ensure that submarkets have the minimum number of items
  while MinItemCount<MinItems
    %Assign items to clusters and ensure that at least the minimum
    Clusters1=floor(rand(NoItems,1).*NoClusters)+1;
    %Calculate the minimum item count
    MinItemCount=NoItems;
    for i=1:NoClusters
      MinItemCount=min(size(find(Clusters1==i),1),MinItemCount);
    end
  end
  MinItemCount=0;   %Ensure that submarkets have the minimum number of items
  while MinItemCount<MinItems
    %Assign items to clusters and ensure that at least the minimum
    Clusters2=floor(rand(NoItems,1).*NoClusters)+1;
    %Calculate the minimum item count
    MinItemCount=NoItems;
    for i=1:NoClusters
      MinItemCount=min(size(find(Clusters2==i),1),MinItemCount);
    end
  end  
    
  [EmpDist.Rand(iPoint),EmpDist.AdjRand(iPoint)] = RandIndex4(Clusters1,Clusters2);

end

%Sort the distributions and add to empirical dist structure
EmpDist.Rand=sort(EmpDist.Rand,'ascend');
EmpDist.AdjRand=sort(EmpDist.AdjRand,'ascend');
EmpDist.NoPoints=NoPoints;
