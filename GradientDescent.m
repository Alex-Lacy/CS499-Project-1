%{
Function Name: GradientDecent
Inputs: X (matrix of numeric inputs), y (vector of binary outputs)
        stepSize (poisitve value 0-1)
%}

function [weightMatrix]=GradientDescent(X,y,stepSize,maxIterations)
s=size(X);
m=s(1);
weightVector=zeros(s(2),1);
weightMatrix=zeros(s(2),maxIterations);
%First scale the inputs (each column should have mean 0 and variance 1).
% X1=X;
% for i=1:s(2)
%     u=mean(X(:,i));
%     v=std(X(:,i));
%     X1(:,i)=(X1(:,i)-u)/v;
% end
% 
% y1=y; 
% y1(y==0)=-1;

for i=1:maxIterations
    sum=0;
    %compute the gradient given the current weightVector(mean logistic loss)
%     for 1:m
%     l=log(1+exp(-y1(i)*weightVector'*X1(i,:)));
%     sum=sum+l;
%     end
%     mll=sum/m;
    for j=1:m
        gmll=(1/m)*(-y(j)*X(j,:)'./(1+exp(y(j)*(weightVector'*X(j,:)'))));
        sum=sum+gmll;
    end 
    
    %update weightVector by taking a step in the negative gradient direction.
    
    
    weightVector=weightVector-stepSize*sum;
    %store the resulting weightVector in the corresponding column of weightMatrix.
    weightMatrix(:,i)=weightVector;
end
%     sum1=0;
%     for k=1:m
%         weightMatrix';
%         X1(k,:);
%         -y1(k);
%         l=(1/m)*log(1+exp(-y1(k)*weightMatrix'*X1(k,:)'));
%         sum1=sum1+l;
%     end
%     plot(sum1);
end