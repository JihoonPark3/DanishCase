function [Data]=TestSetAnalysis(Data,ResponsesNames, IsPlot)
if nargin<3
    IsPlot=0;
end
NResp=length(Data.RegResult);
if NResp~=length(ResponsesNames)
    error('Wrong!');
end
%% Predict.
Data.TestPredicted=[];

for k=1:NResp
    Data.ErrorAnalysis.TestPredicted(:,k)=predict(Data.RegResult{k}.RegModels, Data.Test.X);
    Data.ErrorAnalysis.TestRMSE(k)=sqrt(mean((Data.Test.Y(:,k)-Data.ErrorAnalysis.TestPredicted(:,k)).^2));
    Data.ErrorAnalysis.Corr(k)=corr(Data.Test.Y(:,k),Data.ErrorAnalysis.TestPredicted(:,k));
    Data.ErrorAnalysis.ErrorsAll(:,k)=Data.Test.Y(:,k)-Data.ErrorAnalysis.TestPredicted(:,k);
end
% Stop for now.....
% Resume 25 Jan 2017.
% 나는 해야만 한다. 

if IsPlot==1
    DisplayAllCrossPlots(Data.Test.Y,Data.ErrorAnalysis.TestPredicted,ResponsesNames,4,40)
end
end