function L = edgelikl(I,scale)

nangs = 16;
angle = 0:360/nangs:360-360/nangs;

[nr,nc] = size(I);

nconvs = length(scale)*length(angle);
D = zeros(nr,nc,nconvs);

count = 0;
for s = scale
    for a = angle
        count = count+1;
        [~,mi] = smorlet(0,s,a,1);
        C = conv2(I,mi,'same');
        C = C.*(C > 0);
        D(:,:,count) = C;
    end
end

M = max(D,[],3);
L = normalize(M);

end