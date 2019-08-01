function DisplayPCAScores(ScoreMat,VarExp,TitleText,MarkerSizeVal)
% The function is to display PC scores
figure;
scatter(ScoreMat(:,1),ScoreMat(:,2),MarkerSizeVal,'filled');
xlabel(sprintf('PC1 (%0.f%%)',VarExp(1)),'Fontsize',16); 
ylabel(sprintf('PC2 (%0.f%%)',VarExp(2)),'Fontsize',16); 
title(TitleText,'Fontsize',16);
axis('equal');
end