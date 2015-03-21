function Qfun = calcQ( A, B, Q, R, K, Sigma, x, u, g )

P = calcP(A,B,Q,R,K,g);
Qfun = x'*Q*x + u'*R*u + g*(A*x+B*u)'*P*(A*x+B*u) + ...
    (g/(1-g))*trace(Sigma*(R+g*B'*P*B));

end