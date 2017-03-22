function [treeBag,featImp,oobPredError] = train(rfFeat,rfLbl,ntrees,minleafsize)
% ntrees = 20; minleafsize = 60;

treeBag = TreeBagger(ntrees,rfFeat,rfLbl,'MinLeafSize',minleafsize,'oobvarimp','on');
if nargout > 1
    featImp = treeBag.OOBPermutedVarDeltaError;
end
if nargout > 2
    oobPredError = oobError(treeBag);
end

end