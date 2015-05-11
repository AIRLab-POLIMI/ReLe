n = 2; m = 5;
A = randn(m,n);
b = rand(m,1)*m;

Np = 1000;

% A = dlmread('A.dat');
% b = dlmread('b.dat');

%% plot polyhedron
close all
P = Polyhedron(A,b);
P.plot('alpha',0.5)
hold on;

%% test different approached
[x_lp, fval_lp, result_lp] = LP_SUMD(A, b);
plot(x_lp(1), x_lp(2), 'sb');

[x_ac, fval_ac, result_ac] = AC(A, b);
plot(x_ac(1), x_ac(2), '*g');

[x_mvie, fval_mvie, result_mvie] = MVIE(A, b);
plot(x_mvie(1), x_mvie(2), 'oy');

[x_cg, fval_cg, result_cg] = CG(A, b, Np);
plot(x_cg(1), x_cg(2), 'xr');

[x_cc, fval_cc, result_cc] = CC(A, b);
plot(x_cc(1), x_cc(2), '+k');

legend ('','LP', 'AC', 'MVIE', 'CG', 'CC')

disp('Ended tests');

return
%% test approximated CG
% 
% options.method='achr';
% tic
% P = cprnd(Np, A, b, options);
% x_cg_mat = mean(P);
% t_matlab = toc;
% tic
% % P = cprnd_mex(A, b, 100);
% dlmwrite('At.dat', A);
% dlmwrite('bt.dat', b);
% system(['./misc/sampling_convex/build/randomwalk At.dat bt.dat ' num2str(Np) ' out.dat']);
% x_cg_sc = dlmread('out.dat')';
% t_systemc = toc;
% plot(x_cg_sc(1),  x_cg_sc(2),  'xr');
% plot(x_cg_mat(1), x_cg_mat(2), '+k');