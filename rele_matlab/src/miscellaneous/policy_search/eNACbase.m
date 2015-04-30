%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Reference: www.scholarpedia.org/article/Policy_gradient_methods
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [w, stepsize] = eNACbase(policy_logdif, dlp, data, gamma, robj, lrate)

F = zeros(dlp+1, dlp+1); % Fisher matrix
g = zeros(dlp+1, 1); % Vanilla gradient
el = zeros(dlp+1, 1); % Elegibility
aR = 0; % Average reward

num_trials = max(size(data));
for trial = 1 : num_trials
	phi = [zeros(dlp,1); 1];
    R = 0; % Cumulated reward

    for step = 1 : max(size(data(trial).a))
		% Derivative of the logarithm of the policy in (s_t, a_t)
		dlogpidtheta = policy_logdif(data(trial).s(:,step), data(trial).a(:,step));

		% Basis functions
 		phi = phi + [gamma^(step - 1) * dlogpidtheta; 0];
		
		% Discounted reward
		R = R + gamma^(step - 1) * data(trial).r(robj,step);
    end
    
    aR = aR + R;
    F = F + phi * phi';
    g = g + phi * R;
    el = el + phi;
end

F = F / num_trials;
g = g / num_trials;
el = el / num_trials;
aR = aR / num_trials;

rankF = rank(F);
if rankF == dlp + 1
    Q = 1 / num_trials * (1 + el' / (num_trials * F - el * el') * el);
    b = Q * (aR - el' / F * g);
    w = F \ (g - el * b);
else
% 	str = sprintf('WARNING: F is lower rank (rank = %d)!!! Should be %d', rankF, dlp+1);
% 	disp(str);
    Q = 1 / num_trials * (1 + el' * pinv(num_trials * F - el * el') * el);
    b = Q * (aR - el' * pinv(F) * g);
    w = pinv(F) * (g - el * b);
end

if nargin >= 6
    lambda = sqrt((g - el * b)' * w / (4 * lrate));
    lambda = max(lambda,1e-8);
    stepsize = 1 / (2 * lambda);
end

w = w(1:end-1);
