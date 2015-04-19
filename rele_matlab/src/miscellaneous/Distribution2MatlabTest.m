clear all;
close all;

delete('/tmp/dist2matlab/samples.dat');
delete('/tmp/dist2matlab/grad.dat');
delete('/tmp/dist2matlab/hess.dat');

addpath(genpath('../Toolbox'));
mkdir('/tmp/dist2matlab');
cppProg = '/home/matteo/Projects/github/ReLe/rele-build/dist2mat';


%distributions: normal, log, chol, diag
dist = 'diag';
dim = 5;
cmd = [cppProg ' ' dist ' ' '/tmp/dist2matlab/p1.dat' ' ' ...
    '/tmp/dist2matlab/p2.dat' ' ' ' ' '/tmp/dist2matlab/point.dat' ];
if strcmp(dist,'normal') == 1
    mean = rand(dim,1);
    A = rand(dim,dim);
    A = triu(A);
    cov = A'*A;
    dlmwrite('/tmp/dist2matlab/p1.dat', mean, 'delimiter', '\t', 'precision', 10);
    dlmwrite('/tmp/dist2matlab/p2.dat', cov, 'delimiter', '\t', 'precision', 10);
    
elseif strcmp(dist,'log') == 1
    mean = rand(dim,1);
    w = rand(dim,1);
    varas = rand(dim,1);
    dlmwrite('/tmp/dist2matlab/p1.dat', mean, 'delimiter', '\t', 'precision', 10);
    dlmwrite('/tmp/dist2matlab/p2.dat', w, 'delimiter', '\t', 'precision', 10);
    dlmwrite('/tmp/dist2matlab/p3.dat', varas, 'delimiter', '\t', 'precision', 10);
    di = constant_logistic_gaussian_policy(dim,mean,w,varas);
    
    
    cmd = [cppProg ' ' dist ' ' '/tmp/dist2matlab/p1.dat' ' ' ...
        '/tmp/dist2matlab/p2.dat ' '/tmp/dist2matlab/p3.dat' ...
        ' /tmp/dist2matlab/point.dat' ];
elseif strcmp(dist,'chol') == 1
    mean = randi(29)*rand(dim,1);
    A = randi(10)*rand(dim,dim);
    A = triu(A);
    cov = A'*A;
    A = chol(cov);
    dlmwrite('/tmp/dist2matlab/p1.dat', mean, 'delimiter', '\t', 'precision', 10);
    dlmwrite('/tmp/dist2matlab/p2.dat', A, 'delimiter', '\t', 'precision', 10);
    di = constant_chol_gaussian_policy(dim,mean,A);
elseif strcmp(dist,'diag') == 1
    mean = randi(29)*rand(dim,1);
    sigma = randi(10)*rand(dim,1);
    dlmwrite('/tmp/dist2matlab/p1.dat', mean, 'delimiter','\t', 'precision', 10);
    dlmwrite('/tmp/dist2matlab/p2.dat', sigma, 'delimiter','\t', 'precision', 10);
    di = constant_diag_gaussian_policy(dim,mean,sigma);
else
    error('Unknown distribution!!');
end

point = rand(dim,1);
dlmwrite('/tmp/dist2matlab/point.dat', point, 'delimiter', '\t', 'precision', 10);
mg = di.dlogPidtheta(point);

disp('----- command output -----');
status = system(cmd);
disp('--------------------------');

samples  = dlmread('/tmp/dist2matlab/samples.dat');
gradient = dlmread('/tmp/dist2matlab/grad.dat');
hessian  = dlmread('/tmp/dist2matlab/hess.dat');
inv(hessian)

[sum(samples,2)/size(samples,2),mean]
for i=1:dim
    dsigma(i) = std(samples(i,:));
end
dsigma
norm(mg-gradient, Inf)
assert(norm(mg-gradient, Inf) < 1e-5);