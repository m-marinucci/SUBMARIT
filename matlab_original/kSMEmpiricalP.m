function [PStruct]=kSMEmpiricalP(EmpDist,Cluster)
%Calculates the p values and residual p values from the empirical
%distributon and passed cluster values
%INPUTS
%A structure "EmpDisp" containing the following
%NoPoint - The number of points in the empirical distribution
%zDist - A NoPoints*1 distribution of the z value of UJH (sorted asc)
%LLDist - A NoPoints*1 distribution of log likelihood values (sorted desc)
%DiffDist - A NoPoints*1 distribution of the sum of (PHat-P) (sorted asc
%A structure "Clusters" that consists of the following
%Assign - A products*1 listing of cluster assignments (from 1 to NoClusters)
%Indexes{i} - For i=1..NoClusters, a no. products in cluster*1 vector of product indexes
%Count{i} - For i=1..NoClusters, a the no. products in cluster
%Diff - sum of (Phat-P)
%ItemDiff - sum of (Phat-P)/No. Items
%ScaledDiff - sum of (PHat-P)/P
%ZValue - The z value as per UJH
%MaxObj - The objective function (same as diff)
%LogLH - The lofg likelihood from the maximum likelihood mode
%OUTPUTS
%A structure "PStruct" containing the following
%Zp - [higherp,lowerp] values for the z value of UJH (sorted asc)
%LLp - [higherp,lowerp] values for the log likelihood values (sorted desc)
%Diffp - [higherp,lowerp] values for the of the sum of (PHat-P) (sorted asc

%Find indexes with higher value than checked value
GrIndexes=find(EmpDist.zDist>Cluster.ZValue);
if isempty(GrIndexes)
  PStruct.Zp=[0,1]; %Result better than all values of empirical distribution
else
  Below=(min(GrIndexes)-1)/EmpDist.NoPoints;
  PStruct.Zp=[1-Below,Below];
end

%As above
GrIndexes=find(EmpDist.DiffDist>Cluster.Diff);
if isempty(GrIndexes)
  PStruct.Diffp=[0,1];
else
  Below=(min(GrIndexes)-1)/EmpDist.NoPoints;
  PStruct.Diffp=[1-Below,Below];
end

%LL is opposite, as a lower value is best and values are descending
GrIndexes=find(EmpDist.LLDist<Cluster.LogLH);
if isempty(GrIndexes)
  PStruct.LLp=[0,1];
else
  Below=(min(GrIndexes)-1)/EmpDist.NoPoints;
  PStruct.LLp=[1-Below,Below];
end

