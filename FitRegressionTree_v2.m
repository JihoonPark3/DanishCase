function Outputs=FitRegressionTree_v2(X,Y,ParamsNames,IndexOfCategorical,ResponseNames,MaxIters,Optimize,nrowPlot)
% Date Modified: 22 Jan 2016
% FitRegressionTree_v2.m
% Include categorical parameter. 



% Deactivate the below for now. This is not important.
% if nargin<6
%     ShowPlot=1;
%     if nargin <5
%         Optimize=1; % If not given, optimize the tree.
%         if nargin <4
%             MaxIters=50;
%         end
%     end
% end

% Get the number of regression models.
NumY=size(Y,2);
if NumY~=length(ResponseNames)
    error('Dim does not match');
end
Outputs=cell(1,size(Y,2));
FigNum=randi(999,1);
figure(FigNum);
ncolPlot=ceil(NumY/nrowPlot);
%% Categorical variables.
CategoricalIndices_input=false(1,length(ParamsNames)); % Default is false. 
for kk=1:length(IndexOfCategorical)
    CategoricalIndices_input(IndexOfCategorical(kk))=true;
end

%%
if Optimize==1
    
    for k=1:size(Y,2)
        Outputs{k}.RegModels=fitrensemble(X,Y(:,k),...
            'CategoricalPredictors', CategoricalIndices_input,'Method',...
            'LSBoost','Learner',templateTree('Surrogate','on'),...
            'OptimizeHyperparameters',{'NumLearningCycles','MaxNumSplits','LearnRate'},...
            'HyperparameterOptimizationOptions',struct('Repartition',true, 'MaxObjectiveEvaluations',MaxIters,...
            'AcquisitionFunctionName','expected-improvement-plus','Kfold',5),'PredictorNames',ParamsNames);
        Outputs{k}.Predicted=predict(Outputs{k}.RegModels,X);
        Outputs{k}.RMSE=mean((Outputs{k}.Predicted-Y(:,k)).^2);Outputs{k}.RMSE=Outputs{k}.RMSE^.5;
        
        figure(FigNum);
        subplot(nrowPlot,ncolPlot,k);
        scatter(Y(:,k),Outputs{k}.Predicted,25,'filled'); xlabel('Training Data','Fontsize',14); ylabel('Predicted','Fontsize',14);hold on;
        a_xlim=get(gca); a_xlim=a_xlim.XLim; ylim(a_xlim); line([a_xlim(1),a_xlim(2)],[a_xlim(1),a_xlim(2)],'LineWidth',3,'Color',[1 0 0]);
        title(ResponseNames{k},'Fontsize',16);    axis('square');
        
    end
    
    
elseif Optimize==0
    FigNum2=111;
    % Implemented on 1 Jan 2017
    % If not optimized, just use the previous method.
    
    for k=1:size(Y,2)
        Outputs{k}.RegModels=fitensemble(X,Y(:,k),...
            'CategoricalPredictors', CategoricalIndices_input,...
            'LSBoost',500, 'Tree','LearnRate',.1, 'PredictorNames',ParamsNames);
        RegErrorCV=crossval(Outputs{k}.RegModels,'kfold',10);
        figure(FigNum2); 
        subplot(nrowPlot,ncolPlot,k);
        plot(kfoldLoss(RegErrorCV,'mode','cumulative'),'LineWidth',3); xlabel('# of trees','Fontsize',16);  ylabel('Cr.Val.Error','Fontsize',16);
        title(ResponseNames{k},'Fontsize',16);    axis('square');
        
        [~,NumTree]=min(kfoldLoss(RegErrorCV,'mode','cumulative'));
        Outputs{k}.RegModels=fitensemble(X,Y(:,k),'LSBoost',NumTree, 'Tree','LearnRate',.1, 'PredictorNames',ParamsNames);
        Outputs{k}.Predicted=predict(Outputs{k}.RegModels,X);
        Outputs{k}.RMSE=mean((Outputs{k}.Predicted-Y(:,k)).^2);Outputs{k}.RMSE=Outputs{k}.RMSE^.5;
        
        figure(FigNum);
        subplot(nrowPlot,ncolPlot,k);
        scatter(Y,Outputs{k}.Predicted,25,'filled'); xlabel('Training Data','Fontsize',14); ylabel('Predicted','Fontsize',14);hold on;
        a_xlim=get(gca); a_xlim=a_xlim.XLim; ylim(a_xlim); line([a_xlim(1),a_xlim(2)],[a_xlim(1),a_xlim(2)],'LineWidth',3,'Color',[1 0 0]);
        title(ResponseNames{k},'Fontsize',16);    axis('square');
        
    end
    
else
    error('Something is wrong');
    
    
end

end