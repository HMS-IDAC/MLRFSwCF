clear, clc

%% load image, model

imIndex = 1;
imPath = sprintf('~/Desktop/Data/TestImages/I%03d.tif',imIndex);
lbPath = sprintf('~/Desktop/Data/TestLabels/L%03d.tif',imIndex);
I = imread(imPath);
gtL = imread(lbPath)+1;

rfModelPath = '~/Desktop/rfModel.mat';
disp('loading rfModel')
tic
load(rfModelPath); % loads rfModel
toc

%% segmentation

I = double(I);
I = I/max(max(I));

[imL,classProbs] = mlrfsImClassify(I,rfModel);

%% watershed post-processing

paramimhmin = 0.1;
parambgthr = 0.1;
M = mlrfsWatershedPostProc(classProbs,paramimhmin,parambgthr);

L = 3*uint8(M);
L(L == 0) = 1;
PredWS = label2rgb(L,'winter');

%% display

gtL(gtL == 4) = 0;
gtL3 = label2rgb(gtL,'winter');
I3 = repmat(uint8(255*I),[1 1 3]);
imL3 = label2rgb(imL,'winter');

figureQSS
subplot(2,2,1), imshow(I3), title('image')
subplot(2,2,2), imshow(gtL3), title('ground truth')
subplot(2,2,3), imshow(imL3), title('rf prediction')
subplot(2,2,4), imshow(PredWS), title('watershed(rf prediction)')