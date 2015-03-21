%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Reference: Jan Peters, Policy Gradient Methods for Control Applications
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

LQR = init_lqr(1);
learnIter = 100;

g = LQR.g;
Q = LQR.Q;
R = LQR.R;
A = LQR.A;
B = LQR.B;
x0 = LQR.x0;
sigma = LQR.Sigma;
% k = unifrnd(-1,0);
k = - 0.5;

lrate = 1;
for i = 1 : learnIter
    P = calcP(A,B,Q,R,k,g);
    w1 = -k' * (R + g * B' * P * B) * sigma^2 - g * A' * P * B * sigma^2;
    w2 = -0.5 * (R + g * B' * P * B) * sigma^3;
    k = k + lrate * w1;
    sigma = sigma + lrate * w2;
    J = 0.5 * calcJ(A,B,Q,R,k,sigma^2,x0,g);
end

[X,L,G] = dare(A,B,Q,R);
error = abs(G) - abs(k)
