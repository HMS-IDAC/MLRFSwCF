function [imL,classProbs] = imclassify(imFeat,treeBag)

[nr,nc,nVariables] = size(imFeat);
rfFeat = reshape(imFeat,[nr*nc,nVariables]);

[~,scores] = predict(treeBag,rfFeat);
[~,indOfMax] = max(scores,[],2);
imL = reshape(indOfMax,[nr,nc]);

classProbs = zeros(nr,nc,size(scores,2));
for i = 1:size(scores,2)
    classProbs(:,:,i) = reshape(scores(:,i),[nr,nc]);
end

end