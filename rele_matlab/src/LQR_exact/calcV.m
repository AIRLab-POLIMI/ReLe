function Vfun = calcV( A, B, Q, R, K, Sigma, x, g )

P = calcP(A,B,Q,R,K,g);
Vfun = x'*P*x + (1/(1-g))*trace(Sigma*(R+g*B'*P*B));

end