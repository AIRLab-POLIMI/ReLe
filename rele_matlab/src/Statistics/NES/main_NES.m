clear all;
fn = '/home/matteo/Projects/github/ReLe/rele-build/deepNesAgentOut_agentData.txt';
fp = fopen(fn,'r');
% stats = ReadPGPEStatistics(fp);
i = 1;
% stats(i) = ReadNESIterationStats(fp)
% i = i + 1;
while ~feof(fp)
    val = ReadNESIterationStats(fp);
    if ~isempty(val)
        stats(i) = val;
        i = i + 1;
    end
end
stats
fclose(fp);



%% test NES

for y = 1:size(stats,2)
    
    J = [stats(y).policies(:).J]';
    Theta = [stats(y).policies(:).policy];
    
    domain = 'deep';
    robj = 1;
    [n_obj, pol_low] = settings(domain);
    
    % If the policy has a learnable variance, we don't want to learn it and
    % we make it deterministic
    n_params = size(pol_low.theta,1) - pol_low.dim_variance_params;
    pol_low = pol_low.makeDeterministic;
    
    mu0 = zeros(n_params,1);
    mu0 = stats(y).metaParams(1:n_params);
    sigma0 = 1 * eye(n_params); % change according to the domain
    sigma0 = stats(y).metaParams(n_params+1:end);
    tau = 50 * ones(size(diag(sigma0)));
    
    % pol_high = constant_logistic_gaussian_policy(n_params,mu0,diag(sigma0),tau);
    % pol_high = constant_smart_gaussian_policy(n_params,mu0,sigma0);
    % pol_high = constant_chol_gaussian_policy(n_params,mu0,chol(sigma0));
    pol_high = constant_diag_gaussian_policy(n_params,mu0,sigma0);
    
    N = 10;
    N_MAX = 100;
    
    % solver = REPS_Solver(0.9,N,N_MAX,pol_high);
    solver = NES_Solver(.1,N,N_MAX,pol_high);
    
    [nat_grad, div] = solver.NESbase(J, Theta);
    
%     for i = 1:length(J)
%             disp(i)
%         assert(max(abs(dlogPidtheta(:,i) - stats(y).policies(i).difflog)) <= 1e-6);
%     end
    div = norm(stats(y).metaGradient - nat_grad)
    assert(div < 1e-5);
    
end