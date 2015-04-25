%load dataset read function
addpath(genpath('../../Statistics'));
%locate program
clear all;
reset(symengine);
excmd = '../../../../rele-build/datadiff2mat';

%% REINFORCE
algorithm = 'enacb';


gamma = 0.99;

% policy for dam
stateDim = 1;
actionDim = 1;
s = sym('s', [stateDim, 1]);
a = sym('a', [actionDim, 1]);
syms c1 c2 c3 c4 w1 w2 w3 w4
phi0 = 1;
phi1 = exp(-norm(s - c1)/w1);
phi2 = exp(-norm(s - c2)/w2);
phi3 = exp(-norm(s - c3)/w3);
phi4 = exp(-norm(s - c4)/w4);
phi  = [phi0; phi1; phi2; phi3; phi4];
centers = [0; 50; 120; 160];
widths  = [50; 20; 40; 50];
phi = subs(phi, [c1;c2;c3;c4;w1;w2;w3;w4], [centers;widths]);

theta = sym('theta', [size(phi,1), 1]);
mu = transpose(phi)*theta;
diff = (a - mu);
sigma = sym('sg', [actionDim,1]);
S = diag(sigma.*sigma);
S = subs(S, sigma, 0.1);
pols = (2*pi)^(-actionDim/2) * det(S)^(-1/2) * exp( ...
    -0.5 * transpose(diff) * inv(S) * diff ...
    );

g = transpose(jacobian(log(pols), theta));
h = jacobian(g, theta);


params = [50; -50; 0; 0; 50];
subbsgp = subs(g, theta, params);
subbshp = subs(h, theta, params);
polgradf = matlabFunction(subbsgp);
polhessf = matlabFunction(subbshp);

% write parameters
mkdir('/tmp/ReLe/datadiff2mat/test/')
dlmwrite('/tmp/ReLe/datadiff2mat/test/params.dat', params, 'delimiter', '\t', 'precision', 10);

tcmd = [excmd ' ' algorithm ' /tmp/ReLe/datadiff2mat/test/params.dat' ...
    ' ' num2str(gamma)];

disp('------------------------');
status = system(tcmd);
disp('------------------------');

% read values
redG = dlmread('/tmp/ReLe/datadiff2mat/test/gradient.dat');
if strcmp(algorithm, 'r')
    redH = dlmread('/tmp/ReLe/datadiff2mat/test/hessian.dat');
end

% Read dataset
disp('Reading data trajectories...')
csv = csvread('/tmp/ReLe/datadiff2mat/test/dataset.dat');

disp('Organizing data in episodes...')
clear data;
episodes = readDataset(csv);
for i = 1:length(episodes)
    data(i).s = episodes(i).x(1:end-1,:)';
    data(i).a = episodes(i).u(1:end-1,:)';
    data(i).r = episodes(i).r(1:end-1,:)';
end
clearvars csv


poldifflog = @(s,a) polgradf(a,s);
poldiff2log = @(s,a) polhessf(s);
if strcmp(algorithm, 'r')
    evalG = eREINFORCE(poldifflog, data, gamma, 1);
    evalH = HessianRF(poldifflog, poldiff2log, data, gamma, 1);
elseif strcmp(algorithm, 'rb')
    evalG = eREINFORCEbase(poldifflog, data, gamma, 1);
elseif strcmp(algorithm, 'g')
    evalG = GPOMDP(poldifflog, data, gamma, 1);
elseif strcmp(algorithm, 'gb')
    evalG = GPOMDPbase(poldifflog, length(params), data, gamma, 1);
elseif strcmp(algorithm, 'enacb')
    evalG = eNACbase(poldifflog, length(params), data, gamma, 1);
else
    error('unknown gradient type!');
end

[redG, evalG]
assert(max(abs(redG-evalG)) <= 1e-5);
if strcmp(algorithm, 'r')
    redH, evalH
    assert(max(max(abs(redH-evalH))) <= 1e-3);
end


