function F = imageFeatures(I,sigmas)

F = [];
for sigma = sigmas%[1 2 4 8 16]
    D = zeros(size(I,1),size(I,2),8);
    [D(:,:,1),D(:,:,2),D(:,:,3),D(:,:,4),D(:,:,5),D(:,:,6),D(:,:,7),D(:,:,8)] = derivatives(I,sigma);
    F = cat(3,F,D);
end

end