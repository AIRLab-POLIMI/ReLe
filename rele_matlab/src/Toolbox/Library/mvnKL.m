% Compute the Kullback-Leibler KL(q||p) divergence from multivariate normal
% q (q = N(a,A)) to multivariate normal p (p = N(b,B)).
% q is the new policy, p is the old one.
% Reference: http://en.wikipedia.org/wiki/Multivariate_normal_distribution
function div = mvnKL(a, A, b, B)
    n = length(b);
    
    div = 0.5 * (log(det(B) / det(A)) +  trace(B \ A) + ...
        (b - a)' / B * (b - a)  - n);
end
