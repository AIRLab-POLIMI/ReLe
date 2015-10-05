function [ output_args ] = plotCovarianceEllipsoid( M, C, N )
%PLOTCOVARIANCEELLIPSOID Plot covariance ellipsoid for N stddev
%   Detailed explanation goes here

% For N standard deviations spread of data, the radii of the eliipsoid will
% be given by N*SQRT(eigenvalues).
[U,L] = eig(C);
radii = N*sqrt(diag(L));

% generate data for "unrotated" ellipsoid
[xc,yc,zc] = ellipsoid(0,0,0,radii(1),radii(2),radii(3));

% rotate data with orientation matrix U and center M
a = kron(U(:,1),xc); 
b = kron(U(:,2),yc); 
c = kron(U(:,3),zc);
data = a+b+c; 
n = size(data,2);
x = data(1:n,:)+M(1); 
y = data(n+1:2*n,:)+M(2); 
z = data(2*n+1:end,:)+M(3);

% now plot the rotated ellipse
sc = surf(x,y,z);

end

