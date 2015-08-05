mdp_vars = lqr_mdpvariables();
dim = mdp_vars.dim;
LQR = lqr_environment(dim);

eReward = [0.3;0.7];
Q = 0;
R = 0;
for i = 1:length(LQR.Q)
    Q = Q + eReward(i)*LQR.Q{i};
    R = R + eReward(i)*LQR.R{i};
end

% compute the optimal controller in closed form
P = eye(size(Q,1));
for i = 1 : 100
    K = -LQR.g*inv(R+LQR.g*(LQR.B'*P*LQR.B))*LQR.B'*P*LQR.A;
    P = Q + LQR.g*LQR.A'*P*LQR.A + LQR.g*K'*LQR.B'*P*LQR.A + LQR.g*LQR.A'*P*LQR.B*K + LQR.g*K'*LQR.B'*P*LQR.B*K + K'*R*K;
end
K = -LQR.g*inv(R+LQR.g*LQR.B'*P*LQR.B)*LQR.B'*P*LQR.A;

J = zeros(length(LQR.Q),1);
for i = 1:length(LQR.Q)
    J(i) = -calcJ(LQR.A, LQR.B, LQR.Q{i}, LQR.R{i}, K, LQR.Sigma, LQR.x0, LQR.g);
end

% ./lqr_GIRL enac 100 2 0.3 0.7
% GIRL ENAC
% Episodes: 100
% Rewards: 0.3 0.7 | Params:   -0.4699  -0.7042
% 
% Weights (gnorm):         0        0
%    2.2383  -0.8674
%   -0.8674   0.3362
% 
% Weights (plane):    0.2793   0.7207

% ./lqr_GIRL enac 1000 2 0.3 0.7
% GIRL ENAC
% Episodes: 1000
% Rewards: 0.3 0.7 | Params:   -0.4699  -0.7042
% 
% Weights (gnorm):         0        0
%    2.1029  -0.8845
%   -0.8845   0.3731
% 
% Weights (plane):    0.2963   0.7037




F = getReferenceFront('lqr',1);
close all;
plot(F(:,1),F(:,2));
A =[    2.1029  -0.8845
 -0.8845   0.3731];
% axis([-212 -205 -168 -163]);

B = A + repmat(J,1,size(A,2));

hold on;
plot(J(1),J(2), 'or');


plot(B(1,:)', B(2,:)', '+g');

A =[2.2383  -0.8674
 -0.8674   0.3362];
C = A + repmat(J,1,size(A,2));

plot(C(1,:)', C(2,:)', 'dk');
% plot(B(:,1)', B(:,2)', '+g');

