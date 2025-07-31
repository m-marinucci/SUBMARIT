SUBMARket Identification and Testing (software)

MATLAB (SUBMARIT)
-----------------

Main Functions
--------------
kSMLocalSearch - The main SUBMARIT (submarket clustering function) - This is the quick approximation function, optimizing (PHat-P) where PHat>P
kSMLocalSearch2 - The main SUBMARIT (submarket clustering function) - This function optimizes the log-likelihood directly

Other Functions
---------------

CreateSubstitutionMatrix - Creates an input substitution matrix from sales data

GAPStatisticUniform - Creates GAP statistic like measures for range of number of clusters

kEvaluateClustering - An old version of kSMEvaluateClustering (included as it is still referenced in other functions)

kSMCreateDist - Creates an empirical distribution for a given switching matrix

kSMEmpiricalP - Calculates the p values and residual p values from the empirical distributon and passed cluster values

kSMEntropy - Introduces the entropy based clustering comparison technique

kSMEvaluateClustering - Evaluates an existing clustering using standard (Diff, LL, and Z) quality critera.

kSMLocalSearchConstrained(2) - A constrained version of kSMLocalSearch - Allows some fixed assignments

kSMFold(2) - Performs k-fold validation and compares with empirical distribution

RandCreateDist - Creates an empirical distribution for a given number of items and number of clusters for the Rand index.

RandEmpiricalP - Calculates the p values and residual p values from the empirical distributon and passed values

RandIndex4 - Calculates the similarity between two cluster configurations

RunClusters(2) - Run SUBMARIT multiple times taking the best solution

RunClustersTopk(2) - Run SUBMARIT multiple times and test agreement between top k solutions relative to empirical distributions

