clear, clc

%% load image, model

imIndex = 1;
imPath = sprintf('Data/TestImages/I%03d.tif',imIndex);
lbPath = sprintf('Data/TestLabels_Sampled_3Classes/L%03d.tif',imIndex);

rfModelPath = 'Model/rfModel.mat';
disp('loading rfModel')
tic
load(rfModelPath); % loads rfModel
toc

%% segmentation

I = imread(imPath);

I = double(I);
I = I/max(max(I));

[imL,classProbs] = mlrfsImClassify(I,rfModel);

%% display

gtL = imread(lbPath);

L = zeros(size(gtL));
labels = rfModel.labels;
for lbIndex = 1:rfModel.nLabels
    L(gtL == labels(lbIndex)) = lbIndex;
end

gtL3 = label2rgb(L,'winter','k');
I3 = repmat(uint8(255*I),[1 1 3]);
imL3 = label2rgb(imL,'winter','k');

figureQSS
subplot(1,3,1), imshow(I3), title('image')
subplot(1,3,2), imshow(gtL3), title('ground truth')
subplot(1,3,3), imshow(imL3), title('rf prediction')