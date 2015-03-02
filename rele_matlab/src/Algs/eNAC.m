%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Reference: www.scholarpedia.org/article/Policy_gradient_methods
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function w = eNAC(policy, data, gamma, robj)

%%% Compute gradient
dlp = policy.dlogPidtheta();
A = zeros(dlp+1, dlp+1); % Fisher matrix
b = zeros(dlp+1, 1); % vanilla gradient
num_trials = max(size(data));
parfor trial = 1 : num_trials
	phi = [zeros(dlp,1); 1];
    R = 0;
	df = 1;
    
    for step = 1 : max(size(data(trial).a))
		%%% Compute the derivative of the logarithm of the policy and
		%%% evaluate it in (s_t, a_t)
		dlogpidtheta = policy.dlogPidtheta(data(trial).s(:,step), data(trial).a(:,step));

		%%% Construct basis functions
 		phi = phi + [df * dlogpidtheta; 0];
		
		%%% Update discounted reward
		R = R + df * (data(trial).r(robj,step));
	
		df = df * gamma;
    end
    
    %%% Update matrices A and b
    A = A + phi * phi';
    b = b + phi * R;
end

rankA = rank(A);
if rankA == dlp + 1
	% disp('A is a full rank matrix!!!');
	w = A \ b;
else
	str = sprintf('WARNING: A is lower rank (rank = %d)!!! Should be %d', rankA, dlp+1);
% 	disp(str);
	w = pinv(A) * b;
end
w = w(1:end-1);