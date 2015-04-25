%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Reference: www.scholarpedia.org/article/Policy_gradient_methods
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [w, stepsize] = eNAC(policy_logdif, dlp, data, gamma, robj, lrate)

F = zeros(dlp+1, dlp+1); % Fisher matrix
g = zeros(dlp+1, 1); % Vanilla gradient
num_trials = max(size(data));

parfor trial = 1 : num_trials
	phi = [zeros(dlp,1); 1];
    R = 0;
    
    for step = 1 : max(size(data(trial).a))
		% Derivative of the logarithm of the policy in (s_t, a_t)
		dlogpidtheta = policy_logdif(data(trial).s(:,step), data(trial).a(:,step));

		% Basis functions
 		phi = phi + [gamma^(step - 1) * dlogpidtheta; 0];
		
		% Discounted reward
		R = R + gamma^(step - 1) * (data(trial).r(robj,step));
    end
    
    F = F + phi * phi';
    g = g + phi * R;
end

rankA = rank(F);
if rankA == dlp + 1
	% disp('A is a full rank matrix!!!');
	w = F \ g;
else
% 	str = sprintf('WARNING: A is lower rank (rank = %d)!!! Should be %d', rankA, dlp+1);
% 	disp(str);
	w = pinv(F) * g;
end

if nargin >= 6
    lambda = sqrt(g' * w / (4 * lrate));
    lambda = max(lambda,1e-8);
    stepsize = 1 / (2 * lambda);
end

w = w(1:end-1);
