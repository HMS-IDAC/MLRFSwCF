function [C1,C2] = circlikl(I,radii,sc,nor,thr) % boundary and interior

C1 = zeros(size(I));
C2 = zeros(size(I));
for r = radii
    A = circcentlikl(I,r,sc,nor);
    Cr1 = zeros(size(A));
    Cr2 = zeros(size(A));
    [rs,cs] = find(A > thr);
    for k = 1:length(rs)
        for a = 0:1/r:2*pi-1/r
            x = round(rs(k)+r*cos(a));
            y = round(cs(k)+r*sin(a));
            if x >= 1 && x <= size(I,1) && y >= 1 && y <= size(I,2)
                Cr1(x,y) = Cr1(x,y)+A(rs(k),cs(k));
            end
        end
        rrs = 1:r;
        rrs = rrs(rand(1,r) < 0.5);
        for rr = rrs
            as = 0:1/rr:2*pi-1/rr;
            as = as(rand(1,length(as)) < 0.5);
            for a = as
                x = round(rs(k)+rr*cos(a));
                y = round(cs(k)+rr*sin(a));
                if x >= 1 && x <= size(I,1) && y >= 1 && y <= size(I,2)
                    Cr2(x,y) = Cr2(x,y)+A(rs(k),cs(k));
                end
            end
        end
    end
    C1 = C1+Cr1-A;
    C2 = C2+Cr2;
end
C1 = normalize(C1);
C2 = normalize(filterGauss2D(C2,sc));

end