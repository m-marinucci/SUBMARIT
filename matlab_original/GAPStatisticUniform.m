function [Out] = GAPStatisticUniform(SWM,DataMatrix,Mink,Maxk,MinItems,NoRuns,NoUniform,IsOutput)
%Creates a simple version of the GAP statistic described in Tibshirani,
%Walther, and Hastie (2001) for the SUBMMART procedure
%INPUTS
%SWM - The input switching matrix
%Mink - The minumum number of clusters to test
%Maxk - The maximum number of clusters to test
%MinItmes - The minimum number of items in a single cluster
%NoRuns - The number of optimization runs per solution
%NoUniform - The number of uniform distributed clusters
%IsOutput - True if output cluster information to command window
%OUTPUTS
%Out - a structure containing a series of Nok (Maxk-Mink+1) * 3 result
%matrices.  Each matrix contains the criterion value for the SUBMARIT
%optimization, the average criterion value for the uniform data, and the
%difference. The criterion are as follows
%ResDiff - Differences between PHat and P
%ResSqDiff - Squared Differences between PHat and P
%ResLH - Log-likelihood statistic only counting PHat>P terms
%ResLH2 - The full log-likelihood statistic


  %Find minum and maximum switching values for the matrix based on SWM
  n=size(SWM,1);
  NoCustomers=size(DataMatrix,1);
  NoProducts=size(DataMatrix,2);
  MaxData = full(max(max(DataMatrix)));
  MinData=full(min(min(DataMatrix)));
   
  %Setup scores as cluster number,number of 
  Out.ResDiff=[[1:(Maxk-Mink+1)]',zeros(Maxk-Mink+1,3)];
  Out.ResDiffSq=Out.ResDiff;
  Out.ResLogLH=Out.ResDiff;
  Out.ResLogLH2=Out.ResDiff;
  Out.ResZValue=Out.ResDiff;
  
  BestDiff=-1e10;
  BestDiffSq=-1e10;
  BestZValue=-1e10;
  BestLogLH=-1e10;
  BestLogLH2=-1e10;
  
  for NoClusters=Mink:Maxk
    if IsOutput==true
      sprintf('Cluster number is %d.',NoClusters)
    end
    Clusters=RunClusters(SWM,NoClusters,MinItems,NoRuns);
    Out.ResDiff(NoClusters-Mink+1,2)=Clusters.Diff;
    Out.ResDiffSq(NoClusters-Mink+1,2)=Clusters.DiffSq;
    Out.ResLogLH(NoClusters-Mink+1,2)=Clusters.LogLH;  
    Out.ResLogLH2(NoClusters-Mink+1,2)=Clusters.LogLH2; 
    Out.ResZValue(NoClusters-Mink+1,2)=Clusters.ZValue; 
    SumDiff=0;SumDiffSq=0;SumLogLH=0;SumLogLH2=0;SumZValue=0;
    for j=1:NoUniform
      X=MinData+rand(NoCustomers,NoProducts)*(MaxData-MinData);
      [UFSWM,PIndexes,PCount2] = CreateForcedSwitching2(X,0,0);
      %Ensure that the number of indexes is same size as original data
      if PCount2>n;
        UFSWM=UFSWM(1:n,1:n);
      end
      Clusters=RunClusters(UFSWM,NoClusters,MinItems,NoRuns);
      SumDiff=SumDiff+Clusters.Diff;
      SumDiffSq=SumDiffSq+Clusters.DiffSq;
      SumLogLH=SumLogLH+Clusters.LogLH;
      SumLogLH2=SumLogLH2+Clusters.LogLH2;
      SumZValue=SumZValue+Clusters.ZValue;
    end
    %Find the best values
    Out.ResDiff(NoClusters-Mink+1,3)=SumDiff/NoUniform;
    Out.ResDiffSq(NoClusters-Mink+1,3)=SumDiffSq/NoUniform;
    Out.ResLogLH(NoClusters-Mink+1,3)=SumLogLH/NoUniform;  
    Out.ResLogLH2(NoClusters-Mink+1,3)=SumLogLH2/NoUniform;  
    Out.ResZValue(NoClusters-Mink+1,3)=SumZValue/NoUniform;
    %Calculate the diffs ensure positive is good
    Out.ResDiff(NoClusters-Mink+1,4)=Out.ResDiff(NoClusters-Mink+1,2)-Out.ResDiff(NoClusters-Mink+1,3);
    Out.ResDiffSq(NoClusters-Mink+1,4)=Out.ResDiffSq(NoClusters-Mink+1,2)-Out.ResDiffSq(NoClusters-Mink+1,3);
    Out.ResLogLH(NoClusters-Mink+1,4)=Out.ResLogLH(NoClusters-Mink+1,3)-Out.ResLogLH(NoClusters-Mink+1,2);
    Out.ResLogLH2(NoClusters-Mink+1,4)=Out.ResLogLH2(NoClusters-Mink+1,3)-Out.ResLogLH2(NoClusters-Mink+1,2);
    Out.ResZValue(NoClusters-Mink+1,4)=Out.ResZValue(NoClusters-Mink+1,2)-Out.ResZValue(NoClusters-Mink+1,3);
    
    if Out.ResDiff(NoClusters-Mink+1,4) > BestDiff
      BestDiff=Out.ResDiff(NoClusters-Mink+1,4);
      Out.BestkDiff=NoClusters;
    end
    if Out.ResDiffSq(NoClusters-Mink+1,4) > BestDiffSq
      BestDiffSq=Out.ResDiffSq(NoClusters-Mink+1,4);
      Out.BestkDiff=NoClusters;
    end   
    if Out.ResLogLH(NoClusters-Mink+1,4) > BestLogLH
      BestLogLH=Out.ResLogLH(NoClusters-Mink+1,4);
      Out.BestkLogLH=NoClusters;
    end  
    if Out.ResLogLH2(NoClusters-Mink+1,4) > BestLogLH2
      BestLogLH2=Out.ResLogLH2(NoClusters-Mink+1,4);
      Out.BestkLogLH2=NoClusters;
    end   
    if Out.ResZValue(NoClusters-Mink+1,4) > BestZValue
      BestZValue=Out.ResZValue(NoClusters-Mink+1,4);
      Out.BestkZValue=NoClusters;
    end       
  end

end

