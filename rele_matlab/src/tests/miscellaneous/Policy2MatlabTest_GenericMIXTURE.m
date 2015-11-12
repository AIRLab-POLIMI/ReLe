%load policies
% addpath(genpath('../Toolbox/Library'));
%locate program
clear all;
reset(symengine);
excmd = '../../../../build/test/pol2mat';

%% Multivariate Normal policy with constant covariance
polname = 'genericmix';
stateDim = 2;
actionDim = 2;
nbPol = 3;
s = sym('s', [stateDim, 1]);
a = sym('a', [actionDim, 1]);
alpha = sym('alpha', [nbPol-1, 1]);
preferences = [alpha; 1 - sum(alpha)];
phi = [1;s(end:-1:1)];
if (actionDim == 2)
    c{1} = phi;
    c{2} = phi;
    phi = blkdiag(c{:});
end
nbFeat = size(phi,1);
w = sym('w', [nbPol*nbFeat, 1]);
idx = 1;


S = -10 + rand(actionDim,actionDim) * 10 * 2;
S = round(S'*S,6);
mixpol = 0;

for k = 1:nbPol
    mu = transpose(phi)*w(idx:idx+nbFeat-1);
    idx = idx + nbFeat;
    diff = (a - mu);
    
    pols(k) = (2*pi)^(-actionDim/2) * det(S)^(-1/2) * exp( ...
        -0.5 * transpose(diff) * inv(S) * diff ...
        );
    mixpol = mixpol + preferences(k) * pols(k);
end

% polf = matlabFunction(pols);

g = transpose(jacobian(log(mixpol), [w;alpha]));
h = jacobian(g, [w;alpha]);

if actionDim == 1
    polDeg = 1;
    state = [1.21321; 0.986];
    action = 0.865;
    wVal = [0.5; 0.245;0.3248];
end

if actionDim == 2
    polDeg = 1;
    state = [1.21321;0.956]*10;
    action = [0.865;1.123];
    wVal = [    0.8393
    1.0821
   -0.9947
    1.1023
    0.3530
   -1.0732
   -0.5907
    0.1250
    1.2200
    1.2397
   -0.9130
    1.2549
    1.2191
   -0.0390
    0.8007
   -0.9550
   -0.2086
    1.1086];
    alphaVal = rand(nbPol,1);
    alphaVal = alphaVal / norm(alphaVal,1);
    alphaVal(end) = [];
    alphaVal = [0.4412
                0.1614];
end

% write parameters
mkdir('/tmp/ReLe/pol2mat/test/')
dlmwrite('/tmp/ReLe/pol2mat/test/params.dat', [wVal; alphaVal], 'delimiter', '\t', 'precision', 10);
dlmwrite('/tmp/ReLe/pol2mat/test/state.dat', state, 'delimiter', '\t', 'precision', 10);
dlmwrite('/tmp/ReLe/pol2mat/test/action.dat', action, 'delimiter', '\t', 'precision', 10);
dlmwrite('/tmp/ReLe/pol2mat/test/deg.dat', polDeg, 'delimiter', '\t');
dlmwrite('/tmp/ReLe/pol2mat/test/variace.dat', S, 'delimiter', '\t', 'precision', 10);
dlmwrite('/tmp/ReLe/pol2mat/test/nbpolicies.dat', nbPol, 'delimiter', '\t');

tcmd = [excmd ' ' polname ' /tmp/ReLe/pol2mat/test/params.dat /tmp/ReLe/pol2mat/test/state.dat ' ...
    '/tmp/ReLe/pol2mat/test/action.dat /tmp/ReLe/pol2mat/test/deg.dat ' ...
    ' /tmp/ReLe/pol2mat/test/variace.dat /tmp/ReLe/pol2mat/test/nbpolicies.dat'];

disp('------------------------');
% status = system(tcmd);
disp('------------------------');

%% read values
redD = dlmread('/tmp/ReLe/pol2mat/test/density.dat');
redG = dlmread('/tmp/ReLe/pol2mat/test/grad.dat');
% redH = dlmread('/tmp/ReLe/pol2mat/test/hessian.dat');

% compute using sym engine
evalD = double(subs(mixpol, [s;a;w;alpha], [state;action;wVal;alphaVal]));
evalG = double(subs(g, [s;a;w;alpha], [state;action;wVal;alphaVal]));
evalH = double(subs(h, [s;a;w;alpha], [state;action;wVal;alphaVal]));
% mval = double(subs(mu,[s;w], [state;wVal]));
% Sval = S;
% pval = mvnpdf(action,mval,Sval);
% assert(abs(evalD-pval) <= 1e-9);

% check
[redD, evalD]
assert(abs(redD-evalD) <= 1e-6);

[redG, evalG]
assert(max(abs(redG-evalG)) <= 1e-5);

% redH, evalH
% assert(max(max(abs(redH-evalH))) <= 1e-5);

% samples = dlmread('/tmp/ReLe/pol2mat/test/samples.dat');
% assert(norm(mean(samples)' - mval) <= 0.1);


