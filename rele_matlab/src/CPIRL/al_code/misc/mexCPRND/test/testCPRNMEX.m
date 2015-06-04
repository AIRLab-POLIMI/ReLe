A = dlmread('A.dat');
b = dlmread('b.dat');

%%
% P = Polyhedron(A,b)
% data = P.chebyCenter()
% data.x
% P.plot()

%%
% n = 2; m = 500;
% A = randn(m,n);
% b = rand(m,1)*m;
f = cprnd_mex(1000,A,b)