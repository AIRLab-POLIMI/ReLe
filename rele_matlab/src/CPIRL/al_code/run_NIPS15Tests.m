function [Wout, Eout, MMc_cg, MMc_mvie, MMc_ac, MMc_cc, MMm, MMl, MMp] = run_NIPS15Tests(oo)
% Make the feature value matrix and the transition matrix
F = make_F;
THETA = make_THETA;
[N, K] = size(F);

% Setup the other parameters
GAMMA = 0.9;
nbWEIGHTS = 1;

T = 50;
nbpoints = 1000;
tol_exp = 1e-3;
tol_mdp = 0.00001;

% Select expert's weights
Wout = [];
Eout = [];
% WW = load('nipsWeights.mat');
for i = 1:nbWEIGHTS
    w = randSimplex(3)
%     w = WW.WW(oo,:)';
    Wout = [Wout, w];
    
    % Choose initial features expectations randomly
    VV = rand(N, K);
    VV = sparse(VV);
    [~, E] = opt_policy_and_feat_exp(THETA, F, GAMMA, w, 'uniform', VV, tol_exp);
    Eout = [Eout, E'];
    
    %% run CPIRL
    algs = @MVIE;
    [output] = CPIRLsub(THETA, F, GAMMA, T, E, 'uniform', algs, tol_mdp);
    MMc_mvie = output.featexp';
    algs = @AC;
    [output] = CPIRLsub(THETA, F, GAMMA, T, E, 'uniform', algs, tol_mdp);
    MMc_ac = output.featexp';
    algs = @CC;
    algs = @(A,b) CG(A, b, nbpoints);
    [output] = CPIRLsub(THETA, F, GAMMA, T, E, 'uniform', algs, tol_mdp);
    MMc_cg = output.featexp';
    
    % Run the CPIRL algorithm
%     [PPc, MMc, ITER, TT] = CPIRL(THETA, F, GAMMA, T, E, 'uniform', algs, tol);
    [output] = CPIRLsub(THETA, F, GAMMA, T, E, 'uniform', algs, tol_mdp);
    MMc_cc = output.featexp';
    
    % Run the MWAL algorithm
    [PPm, MMm, ITER, TT, w] = MWAL(THETA, F, GAMMA, T, E, 'uniform', tol_mdp);
    
    % Run the LPAL algorithm
    [Xl, MMl, TT] = LPAL(THETA, F, GAMMA, E, 'uniform');

    % Run the projection algorithm
    [PPp, MMp, ITER, TT] = PROJ(THETA, F, GAMMA, T, E, 'uniform', tol_mdp);
end


end