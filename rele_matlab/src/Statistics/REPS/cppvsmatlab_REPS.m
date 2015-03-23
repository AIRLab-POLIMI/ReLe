%% Script to read REPS statistics
clc;
addpath('..');

%clear old data
clear all;

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
mu0 = zeros(dim_theta,1);
sigma0 = eye(dim_theta); % change according to the domain
pol_high = constant_smart_gaussian_policy(dim_theta,mu0,sigma0);
N = 20;
N_MAX = length(data(1).policies);
epsilon = 0.9;
solver = REPS_Solver(epsilon,N,N_MAX,pol_high);

for it = 1:size(data,2)
    
    disp(['### Iteration ' num2str(it) ' ###']);
    
    J = [data(it).policies.J]';
    for o = 1:length(data(it).policies)
        Theta(:,o) = data(it).policies(o).policy';
    end
    
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
    
    
    if (it ~= size(data,2))
        solver.update(d,Theta);
        mu = vec2mat(solver.policy.theta(1:solver.policy.dim),solver.policy.dim);
        sigma = vec2mat(solver.policy.theta(solver.policy.dim+1:end),solver.policy.dim);
        
        disp('norm inf media:');
        disp(norm(data(it+1).metaParams'-mu, inf));
        assert(norm(data(it+1).metaParams'-mu, inf)<=1e-4);
        disp('max diff cov:')
        disp(max(max(abs(data(it+1).covariance-sigma))));
        assert(max(max(abs(data(it+1).covariance-sigma)))<=1e-4);
    end
    
    disp('val dual REPS');
    disp([solver.dual(eta,J), solver.dual(etacpp,J)]);
end
