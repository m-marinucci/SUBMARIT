function [PStruct]=RandEmpiricalP(RandDist,AdjRandDist,Rand,AdjRand,CreateConfidence)
%Calculates the p values and residual p values from the empirical
%distributon and passed values
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
%PStruct - Structure containing the following:
%Rand - The Rand Index
%AdjRand = The adjusted rand index
%Randp - The p value for the Rand index
%AdjRandp - The p value for the adjusted Rand index
%RandConf,AdjRandConf - Values of Rand from empirical distribution at CI
%[0.005,0.025,0.05,0.25,0.5,0.75,0.95,0.975,0.995]
%Find indexes with higher value than checked value
PStruct.Rand=Rand;
PStruct.AdjRand=AdjRand;
NoPoints=size(RandDist,1);

GrIndexes=find(RandDist>Rand);
if isempty(GrIndexes)
  PStruct.Randp=[0,1]; %Result better than all values of empirical distribution
else
  Below=(min(GrIndexes)-1)/NoPoints;
  PStruct.Randp=[1-Below,Below];
end

%Find indexes with higher value than checked value
GrIndexes=find(AdjRandDist>AdjRand);
if isempty(GrIndexes)
  PStruct.AdjRandp=[0,1]; %Result better than all values of empirical distribution
else
  Below=(min(GrIndexes)-1)/NoPoints;
  PStruct.AdjRandp=[1-Below,Below];
end

if CreateConfidence==true
 %Create distribution values
  PStruct.RandConf=RandDist([round(0.005*NoPoints),round(0.025*NoPoints),round(0.05*NoPoints),round(0.25*NoPoints),round(0.5*NoPoints),round(0.75*NoPoints),round(0.95*NoPoints),round(0.975*NoPoints),round(0.995*NoPoints)]);
  PStruct.AdjRandConf=AdjRandDist([round(0.005*NoPoints),round(0.025*NoPoints),round(0.05*NoPoints),round(0.25*NoPoints),round(0.5*NoPoints),round(0.75*NoPoints),round(0.95*NoPoints),round(0.975*NoPoints),round(0.995*NoPoints)]);
end