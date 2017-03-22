function J = normalize(I)
    J = I-min(min(I));
    J = J/max(max(J));
end