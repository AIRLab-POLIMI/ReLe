close all
clear all

addpath(genpath('../../..'));

phi = load('/tmp/ReLe/nls/GMM/Phi.txt' , '-ascii');
figure(4)
hold on;
%inliers = all(phi > -1000);
plot3(phi(1, :), phi(2, :), phi(3, :), 'ob');



M = [4.5532, 7.7184,   9.6530];
Cc = [0.3490   0.1448  -0.4941;
          0   0.5174   0.1804;
          0        0   0.6076];

C = Cc'*Cc;      
      
      
plot3(M(1), M(2), M(3), 'dr');      

plotCovarianceEllipsoid(M, C, 2);


