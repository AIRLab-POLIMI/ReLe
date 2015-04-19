%load policies
% addpath(genpath('../Toolbox/Library'));
%locate program
clear all;
reset(symengine);
excmd = '../../../rele-build/pol2mat';

%% Normal policy with state-dependent standard deviation 
polname = 'normalstdstate';
stateDim = 1;
actionDim = 1;
s = sym('s', [stateDim, 1]);
a = sym('a', [actionDim, 1]);
phi = [1;s(end:-1:1)];
w = sym('w', [size(phi,1), 1]);
mu = transpose(phi)*w;
diff = (a - mu);

phis = [1;s;s^2];
ws = sym('ws', [size(phis,1), 1]);
sigma = transpose(phis)*ws;
S = diag(sigma.*sigma);
pols = (2*pi)^(-actionDim/2) * det(S)^(-1/2) * exp( ...
    -0.5 * transpose(diff) * inv(S) * diff ...
    );

% polf = matlabFunction(pols);

g = transpose(jacobian(log(pols), w));
h = jacobian(g, w);


polDeg = 1;
polDegStd = 2;
state = [0.9765];
action = 0.865;
wVal = [0.5;0.11234];
wsVal = [0.31; 1.245; 0.911234];

% write parameters
mkdir('/tmp/ReLe/pol2mat/test/')
dlmwrite('/tmp/ReLe/pol2mat/test/params.dat', wVal, 'delimiter', '\t', 'precision', 10);
dlmwrite('/tmp/ReLe/pol2mat/test/state.dat', state, 'delimiter', '\t', 'precision', 10);
dlmwrite('/tmp/ReLe/pol2mat/test/action.dat', action, 'delimiter', '\t', 'precision', 10);
dlmwrite('/tmp/ReLe/pol2mat/test/deg.dat', polDeg, 'delimiter', '\t');
dlmwrite('/tmp/ReLe/pol2mat/test/degstd.dat', polDegStd, 'delimiter', '\t');
dlmwrite('/tmp/ReLe/pol2mat/test/paramsstd.dat', wsVal, 'delimiter', '\t', 'precision', 10);

tcmd = [excmd ' ' polname ' /tmp/ReLe/pol2mat/test/params.dat '... 
    '/tmp/ReLe/pol2mat/test/state.dat ' ...
    '/tmp/ReLe/pol2mat/test/action.dat /tmp/ReLe/pol2mat/test/deg.dat ' ...
    '/tmp/ReLe/pol2mat/test/degstd.dat '...
    '/tmp/ReLe/pol2mat/test/paramsstd.dat'];

disp('------------------------');
status = system(tcmd);
disp('------------------------');

% read values
redD = dlmread('/tmp/ReLe/pol2mat/test/density.dat');
redG = dlmread('/tmp/ReLe/pol2mat/test/grad.dat');
redH = dlmread('/tmp/ReLe/pol2mat/test/hessian.dat');

% compute using sym engine
evalD = double(subs(pols, [s;a;w;ws], [state;action;wVal;wsVal]));
evalG = double(subs(g, [s;a;w;ws], [state;action;wVal;wsVal]));
evalH = double(subs(h, [s;a;w;ws], [state;action;wVal;wsVal]));
mval = double(subs(mu,[s;w], [state;wVal]));
Sval = double(subs(S, [s;a;w;ws], [state;action;wVal;wsVal]));
pval = mvnpdf(action,mval,Sval);
assert(abs(evalD-pval) <= 1e-9);

% check
[redD, evalD]
assert(abs(redD-evalD) <= 1e-6);

[redG, evalG]
assert(max(abs(redG-evalG)) <= 1e-6);

redH, evalH
assert(max(max(abs(redH-evalH))) <= 1e-6);


