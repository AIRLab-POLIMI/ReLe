%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Reference: www.scholarpedia.org/article/Policy_gradient_methods
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [w, drew] = eNAC_IRL_grad(policy, data, gamma, fReward, dfReward)

dlp = policy.dlogPidtheta();
F = zeros(dlp+1, dlp+1); % Fisher matrix
g = 0; % Vanilla gradient
dg = 0;
num_trials = max(size(data));
drew = 0;

parfor trial = 1 : num_trials
	phi = [zeros(dlp,1); 1];
    R = 0;
    dR = 0;
    
    for step = 1 : size(data(trial).a,2)
		% Derivative of the logarithm of the policy in (s_t, a_t)
		dlogpidtheta = policy.dlogPidtheta(data(trial).s(:,step), data(trial).a(:,step));

		% Basis functions
 		phi = phi + [gamma^(step - 1) * dlogpidtheta; 0];
		
		% Discounted reward
        irf = fReward(data(trial).s(:,step), data(trial).a(:,step), data(trial).nexts(:,step));
%         ird = dfReward(data(trial).s(:,step), data(trial).a(:,step), data(trial).nexts(:,step));
		R = R + gamma^(step - 1) * irf;
%         dR = dR + gamma^(step - 1) * ird;
    end
    
    F = F + phi * phi';
    g = g + phi * R;
%     dg = dg + phi * dR;
end

rankF = rank(F);
if rankF == dlp + 1
	w = F \ g;
%     drew = F \ dg;
else
% 	warning('Fisher matrix is lower rank (%d instead of %d).', rankF, dlp+1);
	w = pinv(F) * g;
%     drew = pinv(F) * dg;
end

w = w(1:end-1);
