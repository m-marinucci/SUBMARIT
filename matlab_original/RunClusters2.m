function [BestClusters]=RunClusters2(SWM,NoClusters,MinItems,NoRuns,OutputFlags)
%Runs multiple SUBMARIT runs for a given switching matrix
%Inputs
%SWM - The forced switching matrix to analyze
%NoClusters - The number of clusters to output
%MinItems - The minimum number of items in the cluster
%NoRuns - The number of runs
%OutputFlages - [i,j] Sets MATLAB command window output.
%  i - The job code
%  j - The number of iterations between output
%Outputs
%A structure "BestClusters" that consists of the following
%Assign - A products*1 listing of cluster assignments (from 1 to NoClusters)
%Indexes{i} - For i=1..NoClusters, a no. products in cluster*1 vector of product indexes
%Count{i} - For i=1..NoClusters, a the no. products in cluster
%Diff - sum of (Phat-P)
%ItemDiff - sum of (Phat-P)/No. Items
%ScaledDiff - sum of (PHat-P)/P
%ZValue - The z value as per UJH
%MaxObj - The objective function (same as diff)
%LogLH - The lofg likelihood from the maximum likelihood model
%-------------------------------------------------------------------------
%Version     Author            Date
%   0.10     Stephen France    11/06/2014
  BestLL = 1E7; %Just a large +ve number

  if ~exist('OutputFlags','var')||isempty(OutputFlags)==true
    JobCode = -1;
    OutputFreq = 10;
  else
    JobCode = OutputFlags(1);
    OutputFreq = OutputFlags(2);
  end

  for i=1:NoRuns
    if mod(i,OutputFreq)==0
      Status=[JobCode,i];
    end
    [Clusters]=kSMLocalSearch2(SWM,NoClusters,MinItems);
    if isnan(Clusters.LogLH)
      Out='Found NaN' 
    end
    if Clusters.LogLH<BestLL
      BestClusters=Clusters;
      BestLL=Clusters.LogLH;
    end
  end
