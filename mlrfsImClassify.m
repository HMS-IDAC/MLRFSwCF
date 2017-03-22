function [imL,classProbs] = mlrfsImClassify(I,rfModel)

% layer 1
F = imageFeatures(I,rfModel.sigmas);
fprintf('rf layer 1...'); tic
[imL,classProbs] = imclassify(F,rfModel.treeBags{1});
fprintf('time: %f s\n', toc);

% remaining layers
F = cat(3,F,repmat(zeros(size(I)),[1 1 rfModel.nLabels+rfModel.nProbMapsFeats]));
for layer = 2:rfModel.nLayers
    F(:,:,rfModel.nImageFeatures+1:rfModel.nImageFeatures+rfModel.nLabels) = classProbs;
    F(:,:,rfModel.nImageFeatures+rfModel.nLabels+1:end) = probMapContextFeatures(classProbs,rfModel.offsets,rfModel.pmSigma,rfModel.edgeLikFeatOn,rfModel.probMapELIndex,rfModel.circFeaturesOn,rfModel.probMapCFIndex,rfModel.radiiRange);
    fprintf('rf layer %d...',layer); tic
    [imL,classProbs] = imclassify(F,rfModel.treeBags{layer});
    fprintf('time: %f s\n', toc);
end

end