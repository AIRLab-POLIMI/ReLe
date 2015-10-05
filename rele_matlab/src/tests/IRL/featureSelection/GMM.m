phi = load('/tmp/ReLe/nls/GMM/Phi.txt' , '-ascii');
figure(4)
hold on;
%inliers = all(phi > -1000);
plot3(phi(1, :), phi(2, :), phi(3, :), 'ob');