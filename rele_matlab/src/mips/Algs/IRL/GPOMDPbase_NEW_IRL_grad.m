%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Reference: www.scholarpedia.org/article/Policy_gradient_methods
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [dJdtheta, drewdJ, b_rewfun, b_rewder] = GPOMDPbase_test(policy, data, gamma, fReward, dfReward)


dlp  = policy.dlogPidtheta;
dJdtheta = zeros(dlp,1);
drewdJ = 0;


%% Compute baselines
num_trials = max(size(data));

% compute maximum horizon
actions = cell(1,numel(data));
[actions{:}] = data.a;
lengths = cellfun('length',actions);

% initialize numerator of reward derivative baseline
bnum_rewder = cell(1, max(lengths));
[bnum_rewder{:}] = deal(0);

bnum_rewfun = zeros(dlp, max(lengths));
bden = zeros(dlp, max(lengths));
for trial = 1 : num_trials
    sumdlogPi = zeros(dlp,1);
    num_steps = size(data(trial).a,2);
    
    for st = 1 : num_steps
        sumdlogPi = sumdlogPi + ...
            policy.dlogPidtheta(data(trial).s(:,st), data(trial).a(:,st));
        irf = fReward (data(trial).s(:,st), data(trial).a(:,st), data(trial).nexts(:,st));
        ird = dfReward(data(trial).s(:,st), data(trial).a(:,st), data(trial).nexts(:,st));
        rew  = gamma^(st - 1) * irf;
        drew = gamma^(st - 1) * ird;
        sumdlogPi2 = sumdlogPi .* sumdlogPi;
        bnum_rewfun(:,st) = bnum_rewfun(:, st) + sumdlogPi2 * rew;
        bnum_rewder{st}   = bnum_rewder{st} + sumdlogPi2 * drew;
        bden(:,st) = bden(:, st) + sumdlogPi2;
    end
end
b_rewfun = bnum_rewfun ./ bden;
b_rewfun(isnan(b_rewfun)) = 0; % When 0 / 0

for i = 1:length(bnum_rewder)
    tmp = repmat(bden(:,i), 1, size(bnum_rewder{i},2));
    b_rewder{i} = bnum_rewder{i} ./ tmp;
    b_rewder{i}(isnan(b_rewder{i})) = 0; % When 0 / 0
end

%% Compute gradient
[nr, nc] = size(b_rewder{1});
totstep = 0;
for trial = 1 : num_trials
	sumdlogPi = zeros(dlp,1);
	for st = 1 : size(data(trial).a,2)
        sumdlogPi = sumdlogPi + ...
			policy.dlogPidtheta(data(trial).s(:,st), data(trial).a(:,st));
        irf = fReward (data(trial).s(:,st), data(trial).a(:,st), data(trial).nexts(:,st));
        ird = dfReward(data(trial).s(:,st), data(trial).a(:,st), data(trial).nexts(:,st));
        rew  = gamma^(st - 1) * irf;
        drew = gamma^(st - 1) * ird;
		dJdtheta = dJdtheta + sumdlogPi .* (ones(dlp, 1) * rew - b_rewfun(:,st));
		drewdJ = drewdJ + repmat(sumdlogPi,1,nc) .* (repmat(drew,nr,1) - b_rewder{st});
        totstep = totstep + 1;
	end
end

if gamma == 1
    dJdtheta = dJdtheta / totstep;
    drewdJ = drewdJ / totstep;
else
    dJdtheta = dJdtheta / num_trials;
    drewdJ = drewdJ / num_trials;
end