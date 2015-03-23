%% Script to read REPS statistics
addpath('..');

%clear old data
clear

%% Read data

disp('Reading agent data...')
csv = csvread('/tmp/ReLe/Deep/BBO/Deep_agentData.log');

disp('Organizing data...')

index = 1;
ep = 1;

while(index < size(csv, 1))
    [data(ep), index] = ReadREPSStatistics(csv, index);
    ep = ep + 1;
end

clearvars csv

%% Test with matlab implementation
addpath(genpath('../../Toolbox/'));
Theta = zeros(length(data(1).policies(1).policy), length(data(1).policies));
dim_theta = length(data(1).policies(1).policy);

for it = 1%:size(data,2)
    
    disp(['### Iteration ' num2str(it) ' ###']);
    
    J = [data(it).policies.J]';
    for o = 1:length(data(it).policies)
        Theta(:,o) = data(it).policies(o).policy';
    end
    
    
    mu0 = zeros(dim_theta,1);
    sigma0 = 100 * eye(dim_theta); % change according to the domain
    tau = 50 * ones(size(diag(sigma0)));
    % pol_high = constant_logistic_gaussian_policy(n_params,mu0,diag(sigma0),tau);
    pol_high = constant_smart_gaussian_policy(dim_theta,mu0,sigma0);
    N = 20;
    N_MAX = length(J);
    epsilon = 0.9;
    solver = REPS_Solver(epsilon,N,N_MAX,pol_high);
    [d, divKL, eta]=solver.optimize(J);
    
    dcpp = dlmread(['/tmp/d' num2str(it-1) '.dat']);
    etacpp = data(it).eta;
    disp('etas:');
    disp([eta, etacpp]);
    dcppcomp = exp( (J - max(J)) / etacpp );
    disp('inf-norm weights:');
    disp(norm(d-dcpp,inf));
    disp('inf-norm weights (cpp/cpp recomputed):');
    disp(norm(dcppcomp-dcpp,inf));
    assert(norm(d-dcpp,inf)<= 1e-3);
    assert(abs(eta-etacpp) <= 1e-4);
    
    solver.update(J,Theta);
    
    mu = vec2mat(solver.policy.theta(1:solver.policy.dim),solver.policy.dim);
    sigma = vec2mat(solver.policy.theta(solver.policy.dim+1:end),solver.policy.dim);
    
    [data(it+1).metaParams',mu]
    
    solver.dual(eta,J)
    solver.dual(etacpp,J)
end
