function M = mlrfsWatershedPostProc(classProbs,paramimhmin,parambgthr)

J = filterGauss2D(1-classProbs(:,:,3),2);
J = imhmin(J,paramimhmin);
K = filterGauss2D(classProbs(:,:,1),2);
bg = K > parambgthr;
J(bg) = -Inf;
J(:,1) = -Inf; J(:,end) = -Inf; J(1,:) = -Inf; J(end,:) = -Inf;
W = watershed(J);
M = W > 1;
M = filterByContact(M,bg);

end