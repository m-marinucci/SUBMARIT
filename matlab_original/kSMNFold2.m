function [Output] = kSMNFold2(SWM,NoClusters,MinItems,NoRuns,NFold,NoRandom,MaxFoldRun)
%Performs k-fold validation for SUBMARIT.  For each i=1:NFold,
%takes NoItmes*(NFold-1)/NFold training items and NoItems/NFold test items,
%optimizes the model for training data, adds training items with
%constrained algorithm and gives empirical distributions for the Rand index
%and adjusted Rand index for the addition of training items.
%
%INPUTS
%SWM - A product * product switching matrix
%NoClusters - The number of clusters
%MinItems - The minimum number of items in a subgroup/cluster
%NoRuns - The number of SUBMARIT runs for each optimiziation
%NFold - The number of folds in the data
%NoRandom - The number of random test item additions to build empirical
%distributions
%MaxFoldRun - The number of test fold runs (defaults to NFold)
%OUTPUTS
%A structure "Output" that consists of the following
%AvRand - The average Rand index
%AvAdjRand - The average adjusted Rand index
%AvRandDist - Empirical distribution for the average Rand index
%AvAdjRandDist - Empirical distribution for the average adjusted Rand index

  NoItems=size(SWM,1);
  
  %Get permutation of the columns
  ColPerm=randperm(NoItems);
 
  if ~exist('NoRandom','var')
    NoRandom=0;
  end 
  
  if ~exist('MaxFoldRun','var')
    MaxFoldRun=NFold;
  end    
  
  for i=1:MaxFoldRun
    Start=1+round((i-1)*NoItems/NFold);
    End=round(i*NoItems/NFold);
    TestIndexes=ColPerm(Start:End);
    TrainIndexes=setdiff(1:NoItems,TestIndexes);
    
    %Now there could be a situation where removing ann index gives a 0 row
    %or column
    SubSWM=SWM(TrainIndexes,TrainIndexes);
    ColSum=sum(SubSWM,1)';  %Ensure both are column vectors
    RowSum=sum(SubSWM,2);
    ChkItems=(ColSum==0)|(RowSum==0);
    RemIndexes=TrainIndexes(find(ChkItems));

    if ~isempty(RemIndexes)
      TestIndexes=sort([TestIndexes,RemIndexes]);
      TrainIndexes=setdiff(TrainIndexes,RemIndexes);
      SubSWM=SWM(TrainIndexes,TrainIndexes);
    end
    
    %Now create a clustering using the training Indexes
    Temp=RunClusters2(SubSWM,NoClusters,MinItems,NoRuns);
    %Assign the test items using a constrained version of the algorithm
    CFold.Clusters{i}=RunClustersConstrained2(SWM,NoClusters,MinItems,Temp.Assign,TrainIndexes,TestIndexes,NoRuns);
    if NoRandom>0
      EmpDist.Random{i}=[];
      TempCol=zeros(NoItems,1);
      TempCol(TrainIndexes)=Temp.Assign;
      NoTestIndexes=size(TestIndexes,2);
      for j=1:NoRandom
        TempCol(TestIndexes)=floor(rand(NoTestIndexes,1)*NoClusters)+1;
        EmpDist.Random{i}=[EmpDist.Random{i},TempCol];
      end  
    end
  end
  
  %Given n-fold validation, there are n(n-1)/2 combinations of training
  %sets find rand and adjrand for all of these
  SumRand=0;SumAdjRand=0;
  for i=1:MaxFoldRun-1
    for j=i+1:MaxFoldRun
      [Rand,AdjRand] = RandIndex4(CFold.Clusters{i}.Assign,CFold.Clusters{j}.Assign);
      SumRand=SumRand+Rand;
      SumAdjRand=SumAdjRand+AdjRand;
    end
  end
  Output.AvRand=2*SumRand./(MaxFoldRun*(MaxFoldRun-1));
  Output.AvAdjRand=2*SumAdjRand./(MaxFoldRun*(MaxFoldRun-1));
  
  %Calculate random for matched pairs of samples

  if NoRandom>0
    %Define an empirical distribution of random variable
    AvRandDist=zeros(NoRandom,1);
    AvAdjRandDist=zeros(NoRandom,1);
    for rCount=1:NoRandom
      SumRand=0;SumAdjRand=0;
      for i=1:MaxFoldRun-1
        for j=i+1:MaxFoldRun
          [Rand,AdjRand] = RandIndex4(EmpDist.Random{i}(:,rCount),EmpDist.Random{j}(:,rCount));
          SumRand=SumRand+Rand;
          SumAdjRand=SumAdjRand+AdjRand;
        end
      end
      AvRandDist(rCount)=2*SumRand./(MaxFoldRun*(MaxFoldRun-1));
      AvAdjRandDist(rCount)=2*SumAdjRand./(MaxFoldRun*(MaxFoldRun-1));  
    end
    %Sort the distribution of random variables
    Output.AvRandDist=sort(AvRandDist);
    Output.AvAdjRandDist=sort(AvAdjRandDist);
  end
end

