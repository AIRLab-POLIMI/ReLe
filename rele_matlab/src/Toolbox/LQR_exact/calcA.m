function Afun = calcA( A, B, Q, R, K, Sigma, x, u, g )

Afun = calcQ(A,B,Q,R,K,Sigma,x,u,g) - calcV(A,B,Q,R,K,Sigma,x,g);

end