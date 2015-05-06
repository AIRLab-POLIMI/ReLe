function [PP, MM, ITER, TT, c] = run_MWAL

% Make the feature value matrix and the transition matrix 
F = make_F;
THETA = make_THETA;

% Setup the other parameters
GAMMA = 0.9;
T = 500;
E = [5.25, 4.15, 5];
%E = [7.5, 5, 5];
%E = [9.5, -0.8967, 0];

% algs = @(A,b) CG(A, b, nbpoints);
% algs = @MVIE;
algs = @AC;
% algs = @CC;

% Run the MWAL algorithm
[PP, MM, ITER, TT] = CPIRL(THETA, F, GAMMA, T, E, 'first', algs);



% Write out that policy
write_out_policy(PP(i, :));
