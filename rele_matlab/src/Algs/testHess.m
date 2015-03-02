clear all
domain = 'lqr';
robj = 1;
iter = 5;

[N, pol, episodes, steps, gamma, avg_rew_setting] = settings(domain);

dim = 2;
LQR = init(dim);
g = LQR.g;
A = LQR.A;
B = LQR.B;
Q1 = LQR.Q{1};
R1 = LQR.R{1};
Q2 = LQR.Q{2};
R2 = LQR.R{2};
x0 = LQR.x0;
Sigma = LQR.Sigma;

syms J1 J2 J P1 P2 k1 k2 k3 k4

K = [k1 0; 0 k4];
K_full = [k1 k2; k3 k4];

% simplified closed form for P when A = B = I
P1 = (Q1+K*R1*K)*(eye(dim)-g*(eye(dim)+2*K+K^2))^-1;
P2 = (Q2+K*R2*K)*(eye(dim)-g*(eye(dim)+2*K+K^2))^-1;

J1 = transpose(x0)*P1*x0 + (1/(1-g))*trace(Sigma*(R1+g*transpose(B)*P1*B));
J2 = transpose(x0)*P2*x0 + (1/(1-g))*trace(Sigma*(R2+g*transpose(B)*P2*B));
J = [J1; J2];
J = J(robj);

D_j = transpose(jacobian(J,K_full(:)));
H_j = hessian(J,K_full(:)); % = jacobian(D_j, K_full(:));

clc

for i = 1 : iter

    [ds, uJ, dJ] = collect_samples(domain,episodes,steps,pol,avg_rew_setting,gamma);
    
    disp('-------------------')
    disp(['-------- ' num2str(i) ' --------'])
    disp('-------------------')
    disp('---- hessian')
    hess_est = HessianRF(pol,ds,gamma,robj)
    hess_ex = double(subs(H_j,K_full(:),pol.theta))

%     disp('---- gradient')
%     grad_est_reinf = eREINFORCE(pol, ds, gamma, robj)
%     grad_est_gpomdp = GPOMDP(pol, ds, gamma, robj)
%     tmp = subs(D_j(:),K_full(:),pol.theta);
%     grad_ex = double(tmp(:))
%     
%     disp('---- total reward')
%     j_est = dJ
%     j_ex = double(subs([J1;J2],K(:),pol.theta))
    
    disp('')
    
end