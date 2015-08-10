% clear all
clc
domain = 'lqr';

addpath('../iodata/')

% [n_obj, pol_high] = settings_IRL(domain);

[~, policy, episodes, steps, gamma] = lqr_settings();
% [~, policy, episodes, steps, gamma] = nls_settings();


% Read DATA
disp('Reading data trajectories...')
csv = csvread('/tmp/ReLe/lqr/GIRL/dataset.dat');
% csv = csvread('/tmp/ReLe/nls/GIRL/dataset.dat');
disp('Organizing data in episodes...')
episodes = readDataset(csv);
clearvars csv

% Transform DATA
for i = 1:length(episodes)
    data(i).s = episodes(i).x(1:end-1,:)';
    data(i).a = episodes(i).u(1:end-1,:)';
    data(i).nexts = episodes(i).x(2:end,:)';
end

% adapt policy
k0 = [-0.4699; -0.7042];
policy = gaussian_fixedvar_new(@lqr_basis_pol_mtx, 2, k0, eye(2));
% policy.theta = [6.5178; -2.5994];



%
gamma = 0.9;
solver = GIRL_Solver(data, policy, @lqr_rewardfun, @lqr_rewardderiv, gamma);
% gamma = 0.95;
% solver = GIRL_Solver(data, policy, @nls_rewardfun_g1, @nls_rewardderiv_g1, gamma);

[x, fval] = solver.solve('gb', 1)