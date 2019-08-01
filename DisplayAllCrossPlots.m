function DisplayAllCrossPlots(TrainY,PredictedY,Titles,nPlotRow,MarkerSize)
%set(0, 'DefaultTextInterpreter', 'none');
NRegModels=size(TrainY,2); % # of Regression models.
if size(TrainY,2)~=size(PredictedY,2)
    error('Something is wrong')
end
% 
% if NRegModels~=size(TrainY,2)
%     error('Something is wrong')
% end

nPlotCol=ceil(NRegModels/nPlotRow);

figure;
for k=1:NRegModels
    subplot(nPlotRow,nPlotCol,k)
    scatter(TrainY(:,k),PredictedY(:,k),MarkerSize,'filled'); hold on;
    a_xlim=get(gca); a_xlim=a_xlim.XLim; ylim(a_xlim); %xlabel('Training','Fontsize',14); ylabel('Predicted','Fontsize',14);
    line([a_xlim(1),a_xlim(2)],[a_xlim(1),a_xlim(2)],'LineWidth',3,'Color',[1 0 0]);
    title(Titles{k},'Fontsize',14);
end




end