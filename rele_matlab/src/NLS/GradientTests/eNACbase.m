%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Reference: www.scholarpedia.org/article/Policy_gradient_methods
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function w = eNACbase(policy, data, gamma, robj)


dim   = length(policy.weights);
sdim  = length(data(1).x(1,:));
sigma = policy.stddev;
wnum  = policy.weights;

syms a
w      = sym('w',   [dim,  1]);
s      = sym('s',   [sdim, 1]);
phii    = [s(end:-1:1)];

varsigma = sigma * sum(phii);
pol = 1/(sqrt(2*pi) * (varsigma)) * exp(-0.5*(a-w'*phii)^2/(varsigma)^2);
% pol = 1/(sqrt(2*pi) * (sigma)) * exp(-0.5*(a-w'*phii)^2/(sigma)^2);
% pretty(pol)
% eval(subs(pol, [w; k; phi; a], [wnum; knum; state; action]))

pgrad = gradient(log(pol), w);

% Compute gradient
dlp = dim;
% Fisher matrix
F = zeros(dlp+1, dlp+1);
% Vanilla gradient
g = zeros(dlp+1, 1);
% Elegibility
el = zeros(dlp+1, 1);
% Average reward
aR = 0;

num_trials = max(size(data));
for trial = 1 : num_trials
	phi = [zeros(dlp,1); 1];
    % Cumulated reward
    R = 0;
    % Discount factor
	df = 1;

    for step = 1 : max(size(data(trial).u))-1
		%%% Compute the derivative of the logarithm of the policy and
		%%% Evaluate it in (s_t, a_t)
        dlogpidtheta = eval(subs(pgrad, [w; s; a], [wnum; data(trial).x(step,:)'; data(trial).u(step,:)']));

		%%% Construct basis functions
 		phi = phi + [df * dlogpidtheta; 0];
		
		%%% Update discounted reward
		R = R + df * data(trial).r(step, robj);
	
		df = df * gamma;
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
	str = sprintf('WARNING: F is lower rank (rank = %d)!!! Should be %d', rankF, dlp+1);
% 	disp(str);
    b = 1 / num_trials * (1 + el' * pinv(num_trials * F - el * el') * el) * (aR - el' * pinv(F) * g);
    w = pinv(F) * (g - el * b);
end
% w = w(1:end-1);