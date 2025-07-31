function [Clusters]=kSMEntropy(SWM,NoClusters,MinItems,OptMode,Assign)
%Introduces the method of entropy clustering.  At each stage
%of the algorithm, each item is added to the cluster that best 
%
%INPUTS
%SWM - A product * product switching matrix
%NoClusters - The number of clusters
%MinItems - The minimum number of items in a subgroup/cluster
%OptMode - 1 Optimize ENT
%          2 Optimize ENTNorm
%          3 Optimize ENTNorm2
%OUTPUTS
%A structure "Clusters" that consists of the following
%Assign - A products*1 listing of cluster assignments (from 1 to NoClusters)
%Indexes{i} - For i=1..NoClusters, a no. products in cluster*1 vector of product indexes
%Count{i} - For i=1..NoClusters, a the no. products in cluster
%ENT - The overall cluster entropy
%ENTNorm - The normalized entropy
%ENTNorm2 - The normalized entropy scaled by the number of items in the
%cluster
%ScaledDiff - sum of (PHat-P)/P
%ZValue - The z value as per UJH
%MaxObj - The objective function (same as diff)
%LogLH - The lofg likelihood from the maximum likelihood model

NoItems=size(SWM,1);
%Check the minimum number of items per cluster 
TotalMinItems=max(MinItems.*NoClusters.*2);
if NoItems<TotalMinItems
  ErrMessage='A minimum of 2*MinimumItemsPerCluster*NoClusters Items is reqired.'
  NoClusters
  MinItems
  NoItems
  return;
end

if ~exist('OptMode','var')
  OptMode=1;
end

%Calculate initial random assignments
MinItemCount=0;
Clusters.SWM=SWM;
Sales=sum(SWM,2);
Clusters.NoClusters=NoClusters;
Clusters.NoItems=NoItems;

if ~exist('Assign','var')
  MinItemCount=0;   %Ensure that submarkets have the minimum number of items
  while MinItemCount<MinItems
    %Assign items to clusters and ensure that at least the minimum
    NewAssign=floor(rand(NoItems,1).*NoClusters)+1;
    %Calculate the minimum item count
    MinItemCount=NoItems;
    for i=1:NoClusters
      MinItemCount=min(size(find(NewAssign==i),1),MinItemCount);
    end
  end
  Clusters.Assign=NewAssign;
else
  Clusters.Assign=Assign;
end

%Clusters.Assign=[ones(5,1);2*ones(5,1)];

%Update the current infomration with the assignments
Clusters.ENT=0;
for i=1:NoClusters
  CurIndexes=find(Clusters.Assign==i);
  Clusters.Indexes{i}=CurIndexes;
  Clusters.Count{i}=size(CurIndexes,1);
  Clusters.Sales{i}=Sales(CurIndexes);
  Clusters.AllSales{i}=sum(Clusters.Sales{i},1);
  p=Clusters.Sales{i}./Clusters.AllSales{i};
  pCond=SWM(CurIndexes,CurIndexes)./(Clusters.Sales{i}*ones(1,Clusters.Count{i}));
  Temp=(pCond.*log(pCond)).*(p*ones(1,Clusters.Count{i}));
  HAll=zeros(Clusters.Count{i},Clusters.Count{i});
  IF=isfinite(Temp);
  HAll(IF)=Temp(IF);
  ENTSum=-sum(sum(HAll));
  
  switch OptMode
    case 1
      Clusters.SubCrit{i}=ENTSum;
    case 2
      Clusters.SubCrit{i}=ENTSum./(log(Clusters.Count{i}));
    case 3
      Clusters.SubCrit{i}=Clusters.Count{i}*ENTSum./(log(Clusters.Count{i}));
  end
end

Iter=0;
AsChange=1;
while (AsChange>0)&&(Iter<1000)
  Iter=Iter+1;

  %Determine the +ve change in PHat-P from moving item to each possible new
  %cluster
  OldAssign=Clusters.Assign;
  RandItems=randperm(NoItems);
  for i=1:NoItems
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
      NewSales=sum(SWM(ExIndexes,ExIndexes),2);
      NewAllSales=sum(NewSales,1);
      Newp=NewSales./NewAllSales;
      NewpCond=SWM(ExIndexes,ExIndexes)./(NewSales*ones(1,ExCount));
      Temp=(NewpCond.*log(NewpCond)).*(Newp*ones(1,ExCount));
      HAll=zeros(ExCount,ExCount);
      IF=isfinite(Temp);
      HAll(IF)=Temp(IF);
      ENTSum=-sum(sum(HAll));
      switch OptMode
        case 1
          NewENT=ENTSum;
        case 2
          NewENT=ENTSum./(log(ExCount));
        case 3
          NewENT=ExCount*ENTSum./(log(ExCount));
      end
      if ExCount<Clusters.Count{jClus}
        ObjChange(jClus)=Clusters.SubCrit{jClus}-NewENT;
      else
        ObjChange(jClus)=NewENT-Clusters.SubCrit{jClus};
      end
    end
      
    %Choose the cluster
    [C,NewCluster]=max(ObjChange,[],2);
    if (NewCluster~=OldCluster)&&(Clusters.Count{OldCluster}>MinItems)
      %Sucess!  Swap cluster
      Clusters.Indexes{NewCluster}=sort([Clusters.Indexes{NewCluster};iItem]);
      Clusters.Count{NewCluster}=Clusters.Count{NewCluster}+1;
      Clusters.Indexes{OldCluster}=Clusters.Indexes{OldCluster}(Clusters.Indexes{OldCluster}~=iItem);
      Clusters.Count{OldCluster}=Clusters.Count{OldCluster}-1;
      Clusters.Assign(iItem)=NewCluster;
      %Update the values of the cluster
      Clusters.SubCrit{NewCluster}=Clusters.SubCrit{NewCluster}+ObjChange(NewCluster);
      Clusters.SubCrit{OldCluster}=Clusters.SubCrit{OldCluster}-ObjChange(OldCluster);
    end
  end

  ChAssign=~(OldAssign==Clusters.Assign);

  AsChange=sum(sum(ChAssign));
end

%Calculate the final values of the overall entropy
Clusters.ENT=0;
Clusters.ENTNorm=0;
Clusters.ENTNorm2=0;
Clusters.Iter=Iter;
for i=1:NoClusters
  CurIndexes=find(Clusters.Assign==i);
  Clusters.Indexes{i}=CurIndexes;
  sn=size(CurIndexes,1);
  Clusters.Count{i}=sn;
  
  X=SWM(CurIndexes,CurIndexes);
  %Calculate Market Share
  Sales=sum(X,2);
  AllSales=sum(Sales,1);
  p=Sales./AllSales;
  pCond=X./(Sales*ones(1,sn));
  %plogp is 0 if p is 0 (i.e. log p is - inf)
  Temp=(pCond.*log(pCond)).*(p*ones(1,sn));
  HAll=zeros(sn,sn);
  IF=isfinite(Temp);
  HAll(IF)=Temp(IF);

  ENT=-sum(sum(HAll));
  Clusters.ENT=Clusters.ENT+ENT;
  Clusters.ENTNorm=Clusters.ENTNorm+ENT./(log(sn));
  Clusters.ENTNorm2=Clusters.ENTNorm2+Clusters.Count{i}.*(ENT./(log(sn)));
end

Clusters.ENTNorm=Clusters.ENTNorm./NoClusters;
%Values for ENTNorm are multiplied by no. clusters, so in total n items
Clusters.ENTNorm2=Clusters.ENTNorm2./Clusters.NoItems;

switch OptMode
  case 1
    Clusters.MaxObj=Clusters.ENT;
  case 2
    Clusters.MaxObj=Clusters.ENTNorm;
  case 3
    Clusters.MaxObj=Clusters.ENTNorm2;
end

end


%         %Remove sales for the item
%         NewAllSales=Clusters.AllSales{jClus}-Sales(i);
%         %This should be positive as current pieces are changed
%         ChangeObj(jClus}=Clusters.ENT{i}*Clusters.AllSales{jClus}./NewAllSales;
%         %Now remove values from previous rows and columns
%         RemoveRow=Clusters.p{i}(iItem).*Clusters.Clusters.pCond{i}{iItem,:};
%         RemoveCol=Clusters.p{i}.*Clusters.Clusters.pCond{i}{iItem,:};
%         ChangeObj(jClus)=ChangeObj(jClus)-sum(RemoveRow,2)-sum(RemoveCol,1)+RemoveRow(i);  
%         
%         
%         %Need to count the change by adding variable
%         NewAllSales=Clusters.AllSales{jClus}+Sales(i);
%         %All items are multiplied by p, which is decreased when a new item
%         %is added
%         ChangeObj=Clusters.ENT{i}*Clusters.AllSales{jClus}./NewAllSales;
%         %Now the increase from the new item is the row + column for p(ji)
%         RemoveRow=Clusters.p{i}(iItem).*Clusters.Clusters.pCond{i}{iItem,:};
%         RemoveCol=Clusters.p{i}.*Clusters.Clusters.pCond{i}{iItem,:};
%         ChangeObj(jClus)=ChangeObj(jClus)-sum(RemoveRow,2)-sum(RemoveCol,1)+RemoveRow(i);  
