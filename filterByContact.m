function M = filterByContact(M1,M2)
% M1, M2: masks
% M: objects in M1 that DO NOT touch objects in M2

s = regionprops(M1,M2,'MaxIntensity','PixelIdxList');

mi = cat(1, s.MaxIntensity);

idx = find(mi == 1);

M = M1;
for i = 1:length(idx)
    pidx = s(idx(i)).PixelIdxList;
    M(pidx) = 0;
end

end