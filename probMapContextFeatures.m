function F = probMapContextFeatures(P,offsets,sigma,edgeLikFeatOn,probMapELIndex,circFeaturesOn,probMapCFIndex,radiiRange)
% number of features interferes rftrain and rfsegment

% ----------------------------------------------------------------------
% length(offsets)*8*nLabels features
% ----------------------------------------------------------------------
F = [];
for r = offsets%[5 9 13]
    for a = 0:pi/4:2*pi-pi/4
        v = r*[cos(a) sin(a)];
        T = imtranslate(P,v,'OutputView','same');
        F = cat(3,F,T);
    end
end

% ----------------------------------------------------------------------
% 1 feature
% ----------------------------------------------------------------------
if edgeLikFeatOn
    F = cat(3,F,edgelikl(P(:,:,probMapELIndex),sigma));
end

% ----------------------------------------------------------------------
% 3*2 features
% ----------------------------------------------------------------------
if circFeaturesOn
    Q = P(:,:,probMapCFIndex);
    for r = round(linspace(radiiRange(1),radiiRange(2),3))
        [C1,C2] = circlikl(Q,round(r),sigma,16,0.1);
        F = cat(3,F,C1);
        F = cat(3,F,C2);
    end
end

end