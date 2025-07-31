function [EmpDist]=kSMCreateDist(SWM,NoClusters,NoPoints,MinItems)
%Creates an empirical distribution for a given switching matrix and number
%of clusters.  Does this for each of the switching criteria
%INPUTS
%SWM - A product * product switching matrix
%NoClusters - The number of clusters
%NoPoints - The number of data points in the empirical distribution
%OUTPUTS
%A structure "EmpDisp" containing the following
%NoPoint - The number of points in the empirical distribution
%ZDist - A NoPoints*1 distribution of the z value of UJH (sorted asc)
%LLDist - A NoPoints*1 distribution of log likelihood values (sorted desc)
%DiffDist - A NoPoints*1 distribution of the sum of (PHat-P) (sorted asc)

if ~exist('MinItems','var')
  MinItems=2;
end

NoItems=size(SWM,1);
zDist=[];LLDist=[];DiffDist=[];ENTDist=[];ENTNormDist=[];

for iPoint=1:NoPoints
  %Create a random clustering
  MinItemCount=0;   %Ensure that submarkets have the minimum number of items
  while MinItemCount<MinItems
    %Assign items to clusters and ensure that at least the minimum
    ClusAssign=floor(rand(NoItems,1).*NoClusters)+1;
    %Calculate the minimum item count
    MinItemCount=NoItems;
    for i=1:NoClusters
      MinItemCount=min(size(find(ClusAssign==i),1),MinItemCount);
    end
  end

  [Clusters]=kSMEvaluateClustering(SWM,NoClusters,ClusAssign);
  zDist=[zDist;Clusters.ZValue];
  LLDist=[LLDist;Clusters.LogLH];  
  DiffDist=[DiffDist;Clusters.Diff];
%   Clusters2= CalculateEntropy(SWM,ClusAssign);
%   ENTDist=[ENTDist;Clusters2.ENT];
%   ENTNormDist=[ENTNormDist;Clusters2.ENTNorm];
%   ENTNorm2Dist=[ENTNormDist;Clusters2.ENTNorm2];
end

%Sort the distributions and add to empirical dist structure
EmpDist.zDist=sort(zDist,'ascend');
EmpDist.LLDist=sort(LLDist,'descend');
EmpDist.DiffDist=sort(DiffDist,'ascend');
EmpDist.NoPoints=NoPoints;
