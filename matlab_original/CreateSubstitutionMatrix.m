function [FSWM,PIndexes,PCount] = CreateSubstitutionMatrix(X,Normalize,Weight,Diag)
%Create a forced substitution matrix between products
%For each row i, the columns give the value of other products
%purchased by purchasers of prodjuct i.
%INPUT
%X - A consumer * product substitution matrix
%Normalize - 0 do not normalize (to make row = 1), 1 normalize to make row = 1
%Weight - 0 weight by number of consumers, 1 weight by product sales
%N.B. A value of 1 and 0 is equivalent to the forced substitution matrix given
%in UJH
%Diag - 1 (default) if includes diagonal self substitution, 0  otherwise
%D
%OUTPUT
%FSWM - A produce * product forced substitution matrix
%PIndexes - The indexes of the products that are included
%PCount - The number of products that are included

if ~exist('Diag','var')
  Diag=0;
end

[CCount,PCount]=size(X);
PIndexes=[1:PCount]';

Continue=true;
while Continue==true
  %ensure that consumer has sales for at least two products
  Sorted=sort(X,2,'descend');
  [CurIndexes]=find(Sorted(:,2)>0);
  X=X(CurIndexes,:);

  %Now find products that are bought by at least two consumers
  Sorted=sort(X,1,'descend');
  [CurIndexes]=find(Sorted(2,:)>0);
  X=X(:,CurIndexes);
  PIndexes=PIndexes(CurIndexes);
  
  OldPCount=PCount;
  PCount=size(PIndexes,1);
  Continue=~(PCount==OldPCount);
end
clear Sorted OldPCount

CSales=sum(X,2);
if Weight==0
  X=X./(CSales*ones(1,PCount));
end

FSWM=zeros(PCount,PCount);

for i=1:PCount
  %Calculate ni(1 to number of products)
  if (Weight==0)
    %Weighting by consumers all rows add up to 1
    XMinus=1-(X(:,i)*ones(1,PCount));
  else
    XMinus=(CSales-X(:,i))*ones(1,PCount);
  end
  XProd=X./XMinus;
  Choosei= X(:,i)*ones(1,PCount);
  FSWM(i,:)=sum(XProd.*Choosei,1);
  if Normalize==1
    FSWM(i,:)=FSWM(i,:)./sum(Choosei,1);
  end
end

if (Diag==0)
  FSWM=FSWM-diag(diag(FSWM));
end

end

