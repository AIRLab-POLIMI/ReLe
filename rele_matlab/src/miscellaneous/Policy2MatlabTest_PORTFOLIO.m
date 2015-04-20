%load policies
% addpath(genpath('../Toolbox/Library'));
%locate program
clear all;
reset(symengine);
excmd = '../../../rele-build/pol2mat';

%% Portfolio policy
polname = 'portfolio';
stateDim = 6;
actionDim = 1;
syms epsilon
s = sym('s', [stateDim, 1]);
a = sym('a', [actionDim, 1]);
phi = s;
w = sym('w', [size(phi,1), 1]);
mu = transpose(phi)*w;

pols = epsilon + (1 - 2 * epsilon) * exp(-0.01 * (mu- 10.0)^2);

% polf = matlabFunction(pols);

polDeg = 1;
state = [1.21321; 1.9765; 2.4; 2.3; 1.3; 0.768];
action = 0;
wVal = [0.5; 0.245; 0.11234; 1.3; 4.5; 0.4135];
epsilonVal = 0.05;
assert(action == 1 || action == 0);

if action == 1
    g =  transpose(jacobian(log(pols), w));
else
    g =  transpose(jacobian(log(1-pols), w));
end
h = jacobian(g, w);

% write parameters
mkdir('/tmp/ReLe/pol2mat/test/')
dlmwrite('/tmp/ReLe/pol2mat/test/params.dat', wVal, 'delimiter', '\t', 'precision', 10);
dlmwrite('/tmp/ReLe/pol2mat/test/epsilon.dat', epsilonVal, 'delimiter', '\t', 'precision', 10);
dlmwrite('/tmp/ReLe/pol2mat/test/state.dat', state, 'delimiter', '\t', 'precision', 10);
dlmwrite('/tmp/ReLe/pol2mat/test/action.dat', action, 'delimiter', '\t', 'precision', 10);
dlmwrite('/tmp/ReLe/pol2mat/test/deg.dat', polDeg, 'delimiter', '\t');

tcmd = [excmd ' ' polname ' /tmp/ReLe/pol2mat/test/params.dat '...
    '/tmp/ReLe/pol2mat/test/epsilon.dat '... 
    '/tmp/ReLe/pol2mat/test/state.dat ' ...
    '/tmp/ReLe/pol2mat/test/action.dat '...
    '/tmp/ReLe/pol2mat/test/deg.dat'];

disp('------------------------');
status = system(tcmd);
disp('------------------------');

% read values
redD = dlmread('/tmp/ReLe/pol2mat/test/density.dat');
redG = dlmread('/tmp/ReLe/pol2mat/test/grad.dat');
redH = dlmread('/tmp/ReLe/pol2mat/test/hessian.dat');

% compute using sym engine
evalD = double(subs(pols, [s;a;w;epsilon], [state;action;wVal;epsilonVal]));
if action == 0
    evalD = 1 - evalD;
end
evalG = double(subs(g, [s;a;w;epsilon], [state;action;wVal;epsilonVal]));
evalH = double(subs(h, [s;a;w;epsilon], [state;action;wVal;epsilonVal]));

% check
[redD, evalD]
assert(abs(redD-evalD) <= 1e-6);

[redG, evalG]
assert(max(abs(redG-evalG)) <= 1e-6);

redH, evalH
assert(max(max(abs(redH-evalH))) <= 1e-6);


