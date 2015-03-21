%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Single-objective Natural Gradient optimization for a multidimensional
% LQR.
% The policy is Gaussian with linear mean and constant variance.
% Only the mean is learned.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

N = 2;
LQR = init_lqr(N);
learnIter = 100;

g = LQR.g;
Q = LQR.Q{1};
R = LQR.R{1};
A = LQR.A;
B = LQR.B;
x0 = LQR.x0;
Sigma = eye(N);
K = zeros(N);

for i = 1 : N
%     K(i,i) = unifrnd(-1,0);
    K(i,i) = -0.5;
end

lrate = .1;
for i = 1 : learnIter
    [W1, W2] = calcNatGradient(A,B,Q,R,K,Sigma,g);
    K = K - lrate * W1;
    Sigma = Sigma - lrate * W2;
end

J = calcJ(A,B,Q,R,K,Sigma,x0,g);

[X,L,G] = dare(A,B,Q,R);
error = abs(G) - abs(K)
