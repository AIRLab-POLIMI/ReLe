function [ Mr ] = PFA(M)
%PFA Feature selection with Principal features analysis
%   Given an input dataset, return the features reduced dataset

%Compute covariance
Sigma = cov(M');
 
% Principal component analysis
[A, S] = eig(Sigma);
A = fliplr(A);
S = rot90(S,2);

% At least 90% of variability retained
q = computeDimensions(S, 0.9);


% Select Features
Aq = A(1:q, :);

clusterIndexes = kmeans(Aq',q, 'Replicates', 5);
idx = 1:size(Aq, 2);

selectedFeatures = zeros(q, 1);

for i = 1:q
   clusterVectors = Aq(:, clusterIndexes == i); 
   vectorIdx = idx(clusterIndexes == i);
   clusterMean = sum(clusterVectors, 2)/size(clusterVectors, 2);
   k = dsearchn(clusterVectors', clusterMean');
   selectedFeatures(i) = vectorIdx(k);
end

selectedFeatures = sort(selectedFeatures);

Mr = M(selectedFeatures, :);


end

