function plotPlane( xgv, ygv, pi)
%PLOTPLANE plot a plane
%   TO BE COMPLETED

[X,Y] = meshgrid(xgv, ygv);
a=2; b=-3; c=10; d=-1;
Z=(d- a * X - b * Y)/c;
surf(X,Y,Z)
shading flat

end

