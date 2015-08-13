
addpath('../../mips')
INSTALL()

clear all;
close all;

[x1, x2] = meshgrid(-10:0.1:10, -10:0.1:10);

nbasis = 10;

Sigma = eye(2)*0.1;

mu(1, :) = [0 0];
mu(2, :) = [10 10];

for i=1:nbasis
    angle = (i-1)*2.0/nbasis*pi;
    mu(2+i, :) = [cos(angle), sin(angle)];
end

for i=1:size(mu,1)
    Rbi = sqrt(2*pi)*mvnpdf([x1(:) x2(:)],mu(i, :),Sigma);
    Rb(i, :) = Rbi; 
end

%w = [0.2163 0 0.1544 0 0.6293];
%w = [5.3439e-02   3.9631e-02   2.5759e-02   5.1703e-02   5.2098e-02   5.4219e-02   2.2980e-01   1.2235e-01   1.2512e-01   1.2738e-01   1.1850e-01   3.8334e-10];

w= ones(1, size(mu, 1));
w = w /sum(w);

R = w*Rb;
R = reshape(R,length(x2),length(x1));

figure(1)
surface(x1, x2, R);

[i, j] = find(R == max(R(:)));

p = [x1(i,j),  x2(i,j)];

disp(p)

