function [Clusters]=kSMLocalSearchConstrained2(SWM,NoClusters,MinItems,FixedAssign,AssignIndexes,FreeIndexes)
%Optimizations k submarkets clustering given some possible class
%constraints 
%
%INPUTS
%SWM - A product * product switching matrix
%NoClusters - The number of clusters
%MinItems - The minimum number of items in a subgroup/cluster
%InitialAssign - A product * 1 column vector of the initial assignment of the items to classes
%FreeItems - A column vector of the indices that can be changed
%OUTPUTS
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
%Check the minimum number of items per cluster 
TotalMinItems=max(MinItems.*NoClusters.*2);
if NoItems<TotalMinItems
  ErrMessage='A minimum of 2*MinimumItemsPerCluster*NoClusters Items is reqired.';
  NoClusters
  MinItems
  NoItems
  return;
end

%Calculate assignments
NoAssign=size(AssignIndexes,2);
NoFree=size(FreeIndexes,2);
if size(FixedAssign,1)~=NoAssign
  ErrMessage='Initial Assign needs to contain indexes for each index in the assign indexes vector.';
  return;
end
if NoFree+NoAssign~=NoItems
  ErrMessage='The number of free indexes + assigned indexes should be equal to the number of items.';
  return;
end

%Calculate initial random assignments
MinItemCount=0;
Clusters.SWM=SWM;
Clusters.NoClusters=NoClusters;
Clusters.NoItems=NoItems;
%Get the proportion of switchers and the 
PSales=sum(SWM,2);
PSWM=SWM./(PSales*ones(1,NoItems));
PPSales=PSales./sum(PSales);

InitialAssign=zeros(NoItems,1);
InitialAssign(AssignIndexes)=FixedAssign;
MinItemCount=0;   %Ensure that submarkets have the minimum number of items
while MinItemCount<MinItems
  %Assign items to clusters and ensure that at least the minimum
  InitialAssign(FreeIndexes)=floor(rand(NoFree,1).*NoClusters)+1;
  %Calculate the minimum item count
  MinItemCount=NoItems;
  for i=1:NoClusters
    MinItemCount=min(size(find(InitialAssign==i),1),MinItemCount);
  end
end

Clusters.Assign=InitialAssign;
%Update the current infomration with the assignments
for i=1:NoClusters
  CurIndexes=find(Clusters.Assign==i);
  Clusters.Indexes{i}=CurIndexes;
  Clusters.Count{i}=size(CurIndexes,1);
end

Iter=0;
AsChange=1;
while (AsChange>0)&&(Iter<100)
  Iter=Iter+1;

  %Determine the +ve change in PHat-P from moving item to each possible new
  %cluster
  OldAssign=Clusters.Assign;
  RandItems=FreeIndexes(randperm(NoFree));
  for i=1:NoFree
   iItem=RandItems(i);
    ObjChange=zeros(1,NoClusters);
    for jClus=1:NoClusters
      %Find indexes excluding current index (if in same group
      ExIndexes=Clusters.Indexes{jClus}(Clusters.Indexes{jClus}~=iItem);
      ExCount=size(ExIndexes,1);
      if ExCount==Clusters.Count{jClus}
        %Add index!
        ExIndexes=sort([ExIndexes;iItem]);
        ExCount=ExCount+1;
      else
        OldCluster=jClus;
      end
      SubSWM=PSWM(ExIndexes,ExIndexes);
      %Sum for each item to get PHat value
      PHat(ExIndexes)=sum(SubSWM,2);
      SPPSales=PPSales(ExIndexes);
      Props=ones(ExCount,1)*SPPSales'-diag(SPPSales);
      P(ExIndexes)=sum(Props,2)./(1-SPPSales);
      SVar=P(ExIndexes).*(1-P(ExIndexes)).*PSales(ExIndexes);
      SDComp=log(1/(sqrt( sum(SVar,1)*2*pi)));
      SDiff=sum(PHat(ExIndexes).*PSales(ExIndexes),1)-sum(P(ExIndexes).*PSales(ExIndexes),1);
      NewLogLH=SDComp-(sign(SDiff)*(SDiff.^2))./(2*sum(SVar,1));
      if ExCount<Clusters.Count{jClus}
        ObjChange(jClus)=Clusters.SLogLH{jClus}-NewLogLH;
      else
        ObjChange(jClus)=NewLogLH-Clusters.SLogLH{jClus};
      end         
    end
 
    %Choose the cluster
    [C,NewCluster]=min(ObjChange,[],2);
    if (NewCluster~=OldCluster)&&(Clusters.Count{OldCluster}>MinItems)
      %Sucess!  Swap cluster
      Clusters.Indexes{NewCluster}=sort([Clusters.Indexes{NewCluster};iItem]);
      Clusters.Count{NewCluster}=Clusters.Count{NewCluster}+1;
      Clusters.Indexes{OldCluster}=Clusters.Indexes{OldCluster}(Clusters.Indexes{OldCluster}~=iItem);
      Clusters.Count{OldCluster}=Clusters.Count{OldCluster}-1;
      Clusters.Assign(iItem)=NewCluster;
      %Update the values of the cluster
      Clusters.SLogLH{NewCluster}=Clusters.SLogLH{NewCluster}+ObjChange(NewCluster);
      Clusters.SLogLH{OldCluster}=Clusters.SLogLH{OldCluster}-ObjChange(OldCluster);
    end
  end

  ChAssign=~(OldAssign==Clusters.Assign);

  AsChange=sum(sum(ChAssign));
end

%Setup the P and PHat array
PHat=zeros(NoItems,1);
P=PHat;
Clusters.LogLH=0;
Clusters.LogLH2=0;
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
  Clusters.Var{iClus}=P(Indexes).*(1-P(Indexes)).*PSales(Indexes);
  Clusters.SDComp{iClus}=log(1/(sqrt( sum(Clusters.Var{iClus},1)*2*pi)));
  Clusters.SDiff{iClus}=sum(PHat(Indexes).*PSales(Indexes),1)-sum(P(Indexes).*PSales(Indexes),1);
  Clusters.LogLH=Clusters.LogLH+Clusters.SDComp{iClus}-(sign(Clusters.SDiff{iClus})*(Clusters.SDiff{iClus}.^2))./(2*sum(Clusters.Var{iClus},1));
  Clusters.LogLH2=Clusters.LogLH2+Clusters.SDComp{iClus}-((Clusters.SDiff{iClus}.^2))./(2*sum(Clusters.Var{iClus},1));
end

% Differences <1
Clusters.Diff=sum(PHat-P);
Clusters.DiffSq=sum((PHat-P).^2);
Clusters.ItemDiff=Clusters.Diff./NoItems;
%Scaled Differences
ValidIx=(~isinf(P))&(~isnan(P));
Clusters.ScaledDiff=sum((PHat(ValidIx)-P(ValidIx))./P(ValidIx));
%Z Value as per Urban and Hauser
MPHat=sum(PHat.*PSales);
MP=sum(P.*PSales);
OutVal=(MPHat-MP)./(sum(PHat.*(1-PHat).*PSales).^0.5);
Clusters.ZValue=OutVal;
Clusters.MaxObj=Clusters.Diff;
