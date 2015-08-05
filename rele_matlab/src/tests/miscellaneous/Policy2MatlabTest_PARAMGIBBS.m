%load policies
% addpath(genpath('../Toolbox/Library'));
%locate program
clear all;
reset(symengine);
excmd = '../../../rele-build/pol2mat';

%% Parametric Gibbs Policy
polname = 'paramgibbs';
stateDim = 3;
actionDim = 5;
syms it
s = sym('s', [stateDim, 1]);
a = sym('a', [actionDim, 1]);
phi = [1;a(1);s(end:-1:1)];
w = sym('w', [size(phi,1), 1]);
mu = transpose(phi)*w;

syms den ca
den = 1.0;
for i = 1:actionDim-1
    den = den + exp(it*transpose([1;a(i);s(end:-1:1)])*w);
end

action = ;
if action == actionDim-1
    pols = 1.0 / den;
else
    pols = exp(it*transpose([1;ca;s(end:-1:1)])*w)./ den;
end

% polf = matlabFunction(pols);

polDeg = 1;
state = [1.21321; 1.9765; 2.4];
wVal = [0.5; 0.245; 0.11234; 1.3; 4.5];
inverseT = 1.0;
actions = [0:actionDim-1]';

g =  transpose(jacobian(log(pols), w));
h = jacobian(g, w);

% write parameters
mkdir('/tmp/ReLe/pol2mat/test/')
dlmwrite('/tmp/ReLe/pol2mat/test/params.dat', wVal, 'delimiter', '\t', 'precision', 10);
dlmwrite('/tmp/ReLe/pol2mat/test/inverseT.dat', inverseT, 'delimiter', '\t', 'precision', 10);
dlmwrite('/tmp/ReLe/pol2mat/test/nactions.dat', actionDim, 'delimiter', '\t', 'precision', 10);
dlmwrite('/tmp/ReLe/pol2mat/test/state.dat', state, 'delimiter', '\t', 'precision', 10);
dlmwrite('/tmp/ReLe/pol2mat/test/action.dat', action, 'delimiter', '\t', 'precision', 10);
dlmwrite('/tmp/ReLe/pol2mat/test/deg.dat', polDeg, 'delimiter', '\t');

tcmd = [excmd ' ' polname ' /tmp/ReLe/pol2mat/test/params.dat '...
    '/tmp/ReLe/pol2mat/test/inverseT.dat '... 
    '/tmp/ReLe/pol2mat/test/nactions.dat '...
    '/tmp/ReLe/pol2mat/test/state.dat ' ...
    '/tmp/ReLe/pol2mat/test/action.dat '...
    '/tmp/ReLe/pol2mat/test/deg.dat'];

disp('------------------------');
status = system(tcmd);
disp('------------------------');

% read values
redD = dlmread('/tmp/ReLe/pol2mat/test/density.dat');
redG = dlmread('/tmp/ReLe/pol2mat/test/grad.dat');
% redH = dlmread('/tmp/ReLe/pol2mat/test/hessian.dat');

% compute using sym engine
evalD = double(subs(pols, [s;ca;a;w;it], [state;action;actions;wVal;inverseT]));
evalG = double(subs(g, [s;ca;a;w;it], [state;action;actions;wVal;inverseT]));
evalH = double(subs(h, [s;ca;a;w;it], [state;action;actions;wVal;inverseT]));

% check
[redD, evalD]
assert(abs(redD-evalD) <= 1e-6);

[redG, evalG]
assert(max(abs(redG-evalG)) <= 1e-6);

% redH, evalH
% assert(max(max(abs(redH-evalH))) <= 1e-6);


