clear, clc

%% set parameters

nLayers = 2;
% number of layers in stacked random forest;
% the algorithm will split the training set in nLayers,
% so that every layer sees a similar distribution of the training set;

nLabels = 3;
% number of pixel classes (e.g. background, contour, foreground);
% labels are consecutive integers, starting from 1

nImages = 6; % 60
imPaths = cell(1,nImages);
lbPaths = cell(1,nImages);
for imIndex = 1:nImages
    imPaths{imIndex} = sprintf('~/Desktop/Data/TrainImages/I%03d.tif',imIndex);
    lbPaths{imIndex} = sprintf('~/Desktop/Data/TrainLabels/L%03d.tif',imIndex);
end
% number of images in training set, and paths to images/labels
% edit 'load training set' section below if you want to modify
% the balance of number of labeled pixels per class

sigmas = 2; %[1 2 4 8];
% image features are simply derivatives (up to second order) in different scales;
% this parameter specifies such scales; details in imageFeatures.m

offsets = 5; %[5 9 13];
% in pixels; for offset features from probability maps (see probMapContextFeatures.m)
    
edgeLikFeatOn = true;
% to use edge likelihoods from probability maps (they are computed on the prob map feature of index probMapELIndex (see below))
probMapELIndex = 3;

circFeaturesOn = false;
% circularity features from probability maps (see probMapContextFeatures.m);
% they are computed on the prob map feature of index probMapCFIndex (see below))
probMapCFIndex = 3;

radiiRange = [13 23];
% range of radii on which to compute circularity features

pmSigma = 2;
% used in probMapContextFeatures.m for both edge likelihood features and circularity features

nTrees = 20;
% number of decision trees in the random forest ensemble (in each layer)

minLeafSize = 60;
% minimum number of observations per tree leaf (in each layer)

rfModelFolder = '~/Desktop';
% folder where model (named rfModel.mat) will be saved

%% compute number of features

nImageFeatures = length(sigmas)*8; % see imageFeatures.m
nProbMapsFeats = length(offsets)*8*nLabels+edgeLikFeatOn+circFeaturesOn*6; % see probMapContextFeatures.m

%% load training set

imF = cell(1,nImages);
imL = cell(1,nImages);
for imIndex = 1:nImages
    fprintf('features %d\n',imIndex);

    imPath = imPaths{imIndex};
    lbPath = lbPaths{imIndex};
    
    I = double(imread(imPath));
    I = I/max(max(I));

    imLbl = imread(lbPath)+1; % labels should be 1,2,3,...
    
    bg = imLbl == 1; ct = imLbl == 2; fg = imLbl == 3;
    
    [nr,nc] = size(I);
    ct = ct & rand(nr,nc) < 0.25;
    
    nbg = numel(find(bg)); nct = numel(find(ct)); nfg = numel(find(fg));
    
    fg = fg & rand(nr,nc) < 2*nct/nfg; % balancing fg class, ct assumed to have the smallest amount of samples
    bg = bg & rand(nr,nc) < nct/nbg; % balancing bg class
    
    L = zeros(nr,nc);
    L(bg) = 1;
    L(ct) = 2;
    L(fg) = 3;
    
%     imshow(label2rgb(L,'jet','k','shuffle')), imdistline, pause

    imF{imIndex} = cat(3,imageFeatures(I,sigmas),repmat(zeros(size(I)),[1 1 nLabels+nProbMapsFeats]));
    imL{imIndex} = L;
end

%% split training set

nImagesPerLayer = floor(nImages/nLayers);
layerF = cell(nLayers,nImagesPerLayer);
layerL = cell(nLayers,nImagesPerLayer);
for layer = 1:nLayers
    i0 = (layer-1)*nImagesPerLayer;
    for imIndex = 1:nImagesPerLayer
        layerF{layer,imIndex} = imF{i0+imIndex};
        layerL{layer,imIndex} = imL{i0+imIndex};
%         imshow(label2rgb(layerL{layer,imIndex},'jet','k'))
%         title(sprintf('layer %d, image %d', layer, imIndex))
%         pause
    end
end
clear imF
clear imL

%% train layer 1

ft = [];
lb = [];
for imIndex = 1:nImagesPerLayer
    F = layerF{1,imIndex};
    L = layerL{1,imIndex};
    [rfFeat,rfLbl] = rffeatandlab(F(:,:,1:nImageFeatures),L);
    ft = [ft; rfFeat];
    lb = [lb; rfLbl];
end
fprintf('training layer 1...'); tic
[treeBag,featImp,oobPredError] = train(ft,lb,nTrees,minLeafSize);
figureQSS, subplot(1,2,1), plot(featImp,'o'), title('feature importance, layer 1')
subplot(1,2,2), plot(oobPredError), title('out-of-bag classification error')
fprintf('training time: %f s\n', toc);
save([rfModelFolder '/treeBag1.mat'],'treeBag');
clear treeBag

%% train layers 2...nLayers

for layer = 2:nLayers
    ft = [];
    lb = [];
    for imIndex = 1:nImagesPerLayer
        F = layerF{layer,imIndex};
        L = layerL{layer,imIndex};

        for treeIndex = 1:layer-1
            load([rfModelFolder sprintf('/treeBag%d.mat',treeIndex)]);
            if treeIndex == 1
                [imL,classProbs] = imclassify(F(:,:,1:nImageFeatures),treeBag);
            else
                [imL,classProbs] = imclassify(F,treeBag);
            end
            F(:,:,nImageFeatures+1:nImageFeatures+nLabels) = classProbs;
            F(:,:,nImageFeatures+nLabels+1:end) = probMapContextFeatures(classProbs,offsets,pmSigma,edgeLikFeatOn,probMapELIndex,circFeaturesOn,probMapCFIndex,radiiRange);
        end

        [rfFeat,rfLbl] = rffeatandlab(F,L);
        ft = [ft; rfFeat];
        lb = [lb; rfLbl];
    end

    fprintf('training layer %d...',layer); tic
    [treeBag,featImp,oobPredError] = train(ft,lb,nTrees,minLeafSize);
    figureQSS, subplot(1,2,1), plot(featImp,'o'), title(sprintf('feature importance, layer %d',layer))
    subplot(1,2,2), plot(oobPredError), title('out-of-bag classification error')
    fprintf('training time: %f s\n', toc);
    save([rfModelFolder sprintf('/treeBag%d.mat',layer)],'treeBag');
    clear treeBag
end

%% pack model

rfModel.nLayers = nLayers;
rfModel.nLabels = nLabels;
rfModel.sigmas = sigmas;
rfModel.offsets = offsets;
rfModel.edgeLikFeatOn = edgeLikFeatOn;
rfModel.probMapELIndex = probMapELIndex;
rfModel.circFeaturesOn = circFeaturesOn;
rfModel.probMapCFIndex = probMapCFIndex;
rfModel.radiiRange = radiiRange;
rfModel.pmSigma = pmSigma;
rfModel.nImageFeatures = nImageFeatures;
rfModel.nProbMapsFeats = nProbMapsFeats;
treeBags = cell(1,nLayers);
for i = 1:nLayers
    load([rfModelFolder sprintf('/treeBag%d.mat',i)]);
    treeBags{i} = treeBag;
    delete([rfModelFolder sprintf('/treeBag%d.mat',i)])
end
rfModel.treeBags = treeBags;
disp('saving model')
tic
save([rfModelFolder '/rfModel.mat'],'rfModel','-v7.3');
toc
disp('done training')