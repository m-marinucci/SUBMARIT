function [PStruct]=RunClustersTopk2(SWM,NoClusters,MinItems,NoRuns,Topk,NoRandom)
%Runs multiple SUBMARIT runs for a given switching matrix and gives
%agreement analysis for top k solutions
%Inputs
%SWM - The forced switching matrix to analyze
%NoClusters - The number of clusters to output
%MinItems - The minimum number of items in the cluster
%NoRuns - The number of experimental runs
%Topk - The top k items to be returned
%NoRandom - The number of random values for the empirical distributions
%Outputs
%PStruct - Structure containing the following:
%Rand - The Rand Index
%AdjRand = The adjusted rand index
%Randp - The p value for the Rand index
%AdjRandp - The p value for the adjusted Rand index
%RandConf,AdjRandConf - Values of Rand from empirical distribution at CI
%[0.005,0.025,0.05,0.25,0.5,0.75,0.95,0.975,0.995]
%-------------------------------------------------------------------------
%Version     Author            Date
%   0.10     Stephen France    11/06/2014

  n=size(SWM,1);
  BestLL = ones(1,Topk)*1E7; %Just a large +ve number
  BestAssign = zeros(n,Topk);

  for i=1:NoRuns
    [Clusters]=kSMLocalSearch2(SWM,NoClusters,MinItems);
    if isnan(Clusters.LogLH)
      Out='Found NaN' 
    end
    for j=1:Topk
      if Clusters.LogLH<BestLL(j)
        for k=Topk:-1:j+1
          BestAssign(:,k)=BestAssign(:,k-1);
          BestLL(k)=BestLL(k-1);
        end
        BestAssign(:,j)=Clusters.Assign;
        BestLL(j)=Clusters.LogLH;
        break;
      end
    end
  end
  
  SumRand=0;SumAdjRand=0;
  for i=1:Topk-1
    for j=i+1:Topk
      [Rand,AdjRand] = RandIndex4(BestAssign(:,i),BestAssign(:,j));
      SumRand=SumRand+Rand;
      SumAdjRand=SumAdjRand+AdjRand;
    end
  end
  AvRand=2*SumRand./(Topk*(Topk-1));
  AvAdjRand=2*SumAdjRand./(Topk*(Topk-1));
  
  %Create the values for the random distribution
  AvRandDist=zeros(NoRandom,1);
  AvAdjRandDist=zeros(NoRandom,1); 
  for k=1:NoRandom
    RandAssign=floor(rand(n,Topk)*NoClusters)+1;
    SumRand=0;SumAdjRand=0;
    for i=1:Topk-1
      for j=i+1:Topk
        [Rand,AdjRand] = RandIndex4(RandAssign(:,i),RandAssign(:,j));
        SumRand=SumRand+Rand;
        SumAdjRand=SumAdjRand+AdjRand;
      end
    end
    AvRandDist(k)=2*SumRand./(Topk*(Topk-1));
    AvAdjRandDist(k)=2*SumAdjRand./(Topk*(Topk-1));
  end
  AvRandDist=sort(AvRandDist);
  AvAdjRandDist=sort(AvAdjRandDist);
  
  %Product based measure of reliability
 [PStruct]=RandEmpiricalP(AvRandDist,AvAdjRandDist,AvRand,AvAdjRand,true);

  
  
