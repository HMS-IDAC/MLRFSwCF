function [rfFeat,rfLbl] = rffeatandlab(imFeat,imLbl)

nVariables = size(imFeat,3);

nLabels = max(max(imLbl)); % assuming labels are 1, 2, 3, ...

nPixelsPerLabel = zeros(1,nLabels);
pxlIndices = cell(1,nLabels);

for i = 1:nLabels
    pxlIndices{i} = find(imLbl == i);
    nPixelsPerLabel(i) = numel(pxlIndices{i});
end

nSamples = sum(nPixelsPerLabel);

rfFeat = zeros(nSamples,nVariables);
rfLbl = zeros(nSamples,1);

offset = [0 cumsum(nPixelsPerLabel)];
for i = 1:nVariables
    F = imFeat(:,:,i);
    for j = 1:nLabels
        rfFeat(offset(j)+1:offset(j+1),i) = F(pxlIndices{j});
    end
end
for j = 1:nLabels
    rfLbl(offset(j)+1:offset(j+1)) = j;
end

end