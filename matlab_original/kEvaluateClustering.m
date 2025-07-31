function [Clusters]=kSMEvaluateClustering(SWM,NoClusters,ClusAssign)
%Introduces the method of kSubmarket clustering.  At each stage
%of the algorithm, each item is added to the cluster that best 
%INPUTS
%SWM - A product * product switching matrix
%NoClusters - The number of clusters
%ClusAssign - A products*1 listing of cluster assignments (from 1 to
%NoClusters)
%OUTPUT 
%A structure "Clusters" that consists of the following
%Assign - A products*1 listing of cluster assignments (from 1 to NoClusters)
%Indexes{i} - For i=1..NoClusters, a no. products in cluster*1 vector of product indexes
%Count{i} - For i=1..NoClusters, a the no. products in cluster
%Diff - sum of (Phat-P)
%ItemDiff - sum of (Phat-P)/No. Items
%ScaledDiff - sum of (PHat-P)/P
%ZValue - The z value as per UJH
%MaxObj - The objective function (same as diff)
%LogLH - The lofg likelihood from the maximum likelihood model

NoItems=size(SWM,1);

%Calculate initial random assignments
Clusters.SWM=SWM;
Clusters.NoClusters=NoClusters;
Clusters.NoItems=NoItems;

%Get the proportion of switchers and the 
PSales=sum(SWM,2);
PSWM=SWM./(PSales*ones(1,NoItems));
PPSales=PSales./sum(PSales);

Clusters.Assign=ClusAssign;
%Update the current infomration with the assignments
for i=1:NoClusters
  CurIndexes=find(Clusters.Assign==i);
  Clusters.Indexes{i}=CurIndexes;
  Clusters.Count{i}=size(CurIndexes,1);
end

%Setup the P and PHat array
PHat=zeros(NoItems,1);
P=PHat;
%Calculate the values of the objective function
for iClus=1:NoClusters
  Indexes=Clusters.Indexes{iClus};
  SubSWM=PSWM(Indexes,Indexes);
  %Sum for each item to get PHat value
  PHat(Indexes)=sum(SubSWM,2);
  %Now get the values of P
  %First create matrix of proportions
  SPPSales=PPSales(Indexes);
  Props=ones(Clusters.Count{iClus},1)*SPPSales'-diag(SPPSales);
  P(Indexes)=sum(Props,2)./(1-SPPSales);
end

% Differences <1
Clusters.Diff=sum(PHat-P);
Clusters.ItemDiff=Clusters.Diff./NoItems;
%Scaled Differences
ValidIx=(~isinf(P))&(~isnan(P));
NoValid=sum(ValidIx);
Clusters.ScaledDiff=sum((PHat(ValidIx)-P(ValidIx))./P(ValidIx));
%Z Value as per Urban and Hauser
MPHat=mean(PHat);
MP=mean(P);
AllSales=sum(PSales);
OutVal=(MPHat-MP)./((MP*(1-MP)./AllSales).^0.5);
Clusters.ZValue=OutVal;
Clusters.MaxObj=Clusters.Diff;
%Calculate the value of the maximum(minimum) likelihood function
Var=P(ValidIx).*(1-P(ValidIx))./(PSales(ValidIx));
Part1=NoValid*log(2*pi);
Part2=sum(Var);
Part3=sum(((PHat(ValidIx)-P(ValidIx)).^2)./sqrt(Var));
Clusters.LogLH=-0.5*(Part1+Part2+Part3);
Clusters.P=P';
Clusters.PHat=PHat';
%Calculate the various information criteria



