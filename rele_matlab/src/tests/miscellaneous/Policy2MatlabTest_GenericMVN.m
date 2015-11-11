%load policies
% addpath(genpath('../Toolbox/Library'));
%locate program
clear all;
reset(symengine);
excmd = '../../../../build/test/pol2mat';

%% Multivariate Normal policy with constant covariance
polname = 'genericmvn';
stateDim = 2;
actionDim = 2;
s = sym('s', [stateDim, 1]);
a = sym('a', [actionDim, 1]);
phi = [1;s(end:-1:1)];
if (actionDim == 2)
    c{1} = phi;
    c{2} = phi;
    phi = blkdiag(c{:});
end
w = sym('w', [size(phi,1), 1]);
mu = transpose(phi)*w;
diff = (a - mu);

S = -10 + rand(actionDim,actionDim) * 10 * 2;
S = S'*S;
pols = (2*pi)^(-actionDim/2) * det(S)^(-1/2) * exp( ...
    -0.5 * transpose(diff) * inv(S) * diff ...
    );

% polf = matlabFunction(pols);

g = transpose(jacobian(log(pols), w));
h = jacobian(g, w);

if actionDim == 1
    polDeg = 1;
    state = [1.21321; 0.986];
    action = 0.865;
    wVal = [0.5; 0.245;0.3248];
end

if actionDim == 2
    polDeg = 1;
    state = [1.21321;0.956];
    action = [0.865;1.123];
    wVal = [1.5; 3.245; 0.99;0.003; 0.3245; 0.599];
end

% write parameters
mkdir('/tmp/ReLe/pol2mat/test/')
dlmwrite('/tmp/ReLe/pol2mat/test/params.dat', wVal, 'delimiter', '\t', 'precision', 10);
dlmwrite('/tmp/ReLe/pol2mat/test/state.dat', state, 'delimiter', '\t', 'precision', 10);
dlmwrite('/tmp/ReLe/pol2mat/test/action.dat', action, 'delimiter', '\t', 'precision', 10);
dlmwrite('/tmp/ReLe/pol2mat/test/deg.dat', polDeg, 'delimiter', '\t');
dlmwrite('/tmp/ReLe/pol2mat/test/variace.dat', S, 'delimiter', '\t');

tcmd = [excmd ' ' polname ' /tmp/ReLe/pol2mat/test/params.dat /tmp/ReLe/pol2mat/test/state.dat ' ...
    '/tmp/ReLe/pol2mat/test/action.dat /tmp/ReLe/pol2mat/test/deg.dat ' ...
    ' /tmp/ReLe/pol2mat/test/variace.dat'];

disp('------------------------');
% status = system(tcmd);
disp('------------------------');

%% read values
redD = dlmread('/tmp/ReLe/pol2mat/test/density.dat');
redG = dlmread('/tmp/ReLe/pol2mat/test/grad.dat');
% redH = dlmread('/tmp/ReLe/pol2mat/test/hessian.dat');

% compute using sym engine
evalD = double(subs(pols, [s;a;w], [state;action;wVal]));
evalG = double(subs(g, [s;a;w], [state;action;wVal]));
evalH = double(subs(h, [s;a;w], [state;action;wVal]));
mval = double(subs(mu,[s;w], [state;wVal]));
Sval = S;
pval = mvnpdf(action,mval,Sval);
assert(abs(evalD-pval) <= 1e-9);

% check
[redD, evalD]
assert(abs(redD-evalD) <= 1e-6);

[redG, evalG]
assert(max(abs(redG-evalG)) <= 1e-5);

% redH, evalH
% assert(max(max(abs(redH-evalH))) <= 1e-5);

samples = dlmread('/tmp/ReLe/pol2mat/test/samples.dat');
assert(norm(mean(samples)' - mval) <= 0.1);


