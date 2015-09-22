function [ Mr ] = PCA( M )
%PCA Summary of this function goes here
%   Detailed explanation goes here


sigma = cov(M');
[U, S, ~] = svd(sigma);

q = computeDimensions(S, 0.9);


Mmean = repmat(sum(M, 2)/size(M, 2), 1, size(M, 2));

Mnormalized = M - Mmean;

Z = Mnormalized'*U(:, 1:q) + Mmean'*U(:, 1:q); %TODO check

Mr = Z';

end

