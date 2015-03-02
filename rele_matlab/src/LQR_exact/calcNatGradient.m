function [nat_grad_mean, nat_grad_var] = ...
    calcNatGradient(A, B, Q, R, K, Sigma, g)

P = calcP(A,B,Q,R,K,g);

nat_grad_mean = 2 * K' * (R + g * B' * P *B) * Sigma + ...
    2 * g * A' * P * B * Sigma;

nat_grad_var = 2 * Sigma * (R + g * B' * P * B) * Sigma;

end