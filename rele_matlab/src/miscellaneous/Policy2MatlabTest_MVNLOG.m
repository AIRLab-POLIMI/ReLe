%load policies
% addpath(genpath('../Toolbox/Library'));
%locate program
clear all;
reset(symengine);
excmd = '../../../rele-build/pol2mat';

%% Multivariate Normal policy with diagonal covariance (logistic parameters)
polname = 'mvnlog';
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

asv = sym('asv', [actionDim,1]);
sigma = sym('sg', [actionDim,1]);
for i = 1:actionDim
    S(i,i) = asv(i)/ (1 + exp(-sigma(i)));
end
pols = (2*pi)^(-actionDim/2) * det(S)^(-1/2) * exp( ...
    -0.5 * transpose(diff) * inv(S) * diff ...
    );

% polf = matlabFunction(pols);

g = transpose(jacobian(log(pols), [w;sigma]));
h = jacobian(g, [w;sigma]);

if actionDim == 1
    polDeg = 1;
    state = [1.21321; 0.986];
    action = 0.865;
    wVal = [0.5; 0.245;0.3248];
    sigmaVal = 1.3;
    asvVal = [11.5];
end

if actionDim == 2
    polDeg = 1;
    state = [1.21321;0.956];
    action = [0.865;1.123];
    wVal = [0.5; 0.245; 0.99;0.3; 0.3245; 0.599];
    sigmaVal = [1.3; 0.9];
    asvVal = [13.4;21.5];
end

% write parameters
mkdir('/tmp/ReLe/pol2mat/test/')
dlmwrite('/tmp/ReLe/pol2mat/test/state.dat', state, 'delimiter', '\t', 'precision', 10);
dlmwrite('/tmp/ReLe/pol2mat/test/action.dat', action, 'delimiter', '\t', 'precision', 10);
dlmwrite('/tmp/ReLe/pol2mat/test/params.dat', [wVal;sigmaVal], 'delimiter', '\t', 'precision', 10);
dlmwrite('/tmp/ReLe/pol2mat/test/deg.dat', polDeg, 'delimiter', '\t');
dlmwrite('/tmp/ReLe/pol2mat/test/asvariace.dat', asvVal, 'delimiter', '\t');

tcmd = [excmd ' ' polname ' /tmp/ReLe/pol2mat/test/params.dat /tmp/ReLe/pol2mat/test/state.dat ' ...
    '/tmp/ReLe/pol2mat/test/action.dat /tmp/ReLe/pol2mat/test/deg.dat ' ...
    ' /tmp/ReLe/pol2mat/test/asvariace.dat'];

disp('------------------------');
status = system(tcmd);
disp('------------------------');

% read values
redD = dlmread('/tmp/ReLe/pol2mat/test/density.dat');
redG = dlmread('/tmp/ReLe/pol2mat/test/grad.dat');
redH = dlmread('/tmp/ReLe/pol2mat/test/hessian.dat');

% compute using sym engine
evalD = double(subs(pols, [s;a;w;sigma;asv], [state;action;wVal;sigmaVal;asvVal]));
evalG = double(subs(g, [s;a;w;sigma;asv], [state;action;wVal;sigmaVal;asvVal]));
evalH = double(subs(h, [s;a;w;sigma;asv], [state;action;wVal;sigmaVal;asvVal]));
mval = double(subs(mu,[s;w], [state;wVal]));
Sval = double(subs(S, [sigma;asv], [sigmaVal;asvVal]));
pval = mvnpdf(action,mval,Sval);
assert(abs(evalD-pval) <= 1e-9);

% check
[redD, evalD]
assert(abs(redD-evalD) <= 1e-6);

[redG, evalG]
assert(max(abs(redG-evalG)) <= 1e-6);

redH, evalH
assert(max(max(abs(redH-evalH))) <= 1e-6);


