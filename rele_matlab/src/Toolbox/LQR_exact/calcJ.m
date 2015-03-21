function J = calcJ( A, B, Q, R, K, Sigma, x0, g )

P = calcP(A,B,Q,R,K,g);
J = x0'*P*x0 + (1/(1-g))*trace(Sigma*(R+g*B'*P*B));

end