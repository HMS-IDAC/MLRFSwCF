clear, clc

%% set parameters

nLayers = 2;
% number of layers in stacked random forest;
% the algorithm will split the training set in nLayers,
% so that every layer sees a similar distribution of the training set;
% should be an integer >= 1; values > 5 are not recommended

labels = [1 2 3];
% class labels present in the annotation/label images
% example: if your label images encode label 1, 2, 3, for (respectively) background,
% contour, foreground, you'd set labels = [1 2 3]; you can also
% simply set labels = [1 3] if you only want to detect background and foreground

nImages = 2; % 60
imPaths = cell(1,nImages);
lbPaths = cell(1,nImages);
for imIndex = 1:nImages
    imPaths{imIndex} = sprintf('Data/TrainImages/I%03d.tif',imIndex);
    lbPaths{imIndex} = sprintf('Data/TrainLabels_Sampled_3Classes/L%03d.tif',imIndex);
end
% number of images in training set, and paths to images/labels

sigmas = 2; %[1 2 4 8];
% image features are simply derivatives (up to second order) in different scales;
% this parameter specifies such scales; details in imageFeatures.m

offsets = 5; %[5 9 13];
% in pixels; for offset features from probability maps (see probMapContextFeatures.m)
    
edgeLikFeatOn = true;
% to use edge likelihoods from probability maps
probMapELIndex = 2;
% edge likelihoods are computed on the prob map feature of index probMapELIndex;
% example: if labels = [1 3] and you want to compute edge likelihoods
% from label 3, you'd set probMapELIndex = 2, because 3 is at index 2 in labels;
% use-case: this will typically correspond to your 'foreground' class, which
% can be 'nuclei' if you're trying to separate nuclei from background

circFeaturesOn = true;
% circularity features from probability maps (see probMapContextFeatures.m);
probMapCFIndex = 2;
% circularity features are computed on the prob map feature of index probMapCFIndex;
% example: if labels = [1 3] and you want to compute circularity features
% from label 3, you'd set probMapCFIndex = 2, because 3 is at index 2 in labels;
% use-case: this will typically correspond to the class containing round
% objects -- which can be 'nuclei' if you're trying to separate nuclei from background;
% it is recommended to use circularity features in such cases,
% since performance tends to improve

radiiRange = [12 24];
% range of radii on which to compute circularity features

pmSigma = 2;
% used in probMapContextFeatures.m for both edge likelihood features and circularity features

nTrees = 20;
% number of decision trees in the random forest ensemble (in each layer)

minLeafSize = 60;
% minimum number of observations per tree leaf (in each layer)

rfModelFolderPath = 'Model';
% path to folder where model (named rfModel.mat) will be saved

% 
% no parameters to set beyond this point
%

%% compute number of features

nLabels = length(labels);
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

    imLbl = imread(lbPath); % labels should be 1,2,3,...
    L = zeros(size(imLbl));
    for lbIndex = 1:nLabels
        L(imLbl == labels(lbIndex)) = lbIndex;
    end
    
%     imshow(label2rgb(L,'winter','k')), imdistline, pause

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
if exist(rfModelFolderPath,'dir') ~= 7
    mkdir(rfModelFolderPath);
end
save([rfModelFolderPath '/treeBag1.mat'],'treeBag');
clear treeBag

%% train layers 2...nLayers

for layer = 2:nLayers
    ft = [];
    lb = [];
    for imIndex = 1:nImagesPerLayer
        F = layerF{layer,imIndex};
        L = layerL{layer,imIndex};

        for treeIndex = 1:layer-1
            load([rfModelFolderPath sprintf('/treeBag%d.mat',treeIndex)]);
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
    save([rfModelFolderPath sprintf('/treeBag%d.mat',layer)],'treeBag');
    clear treeBag
end

%% pack model

rfModel.nLayers = nLayers;
rfModel.labels = labels;
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
    load([rfModelFolderPath sprintf('/treeBag%d.mat',i)]);
    treeBags{i} = treeBag;
    delete([rfModelFolderPath sprintf('/treeBag%d.mat',i)])
end
rfModel.treeBags = treeBags;
disp('saving model')
tic
save([rfModelFolderPath '/rfModel.mat'],'rfModel','-v7.3');
toc
disp('done training')