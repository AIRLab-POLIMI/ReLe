mkdir('/tmp/ReLe/chol_FIM');
mkdir('/tmp/ReLe/chol_FIM/test');
cmd = '/home/matteo/Projects/github/ReLe/rele-build/src/test/chol_FIM';

n_params = 4;
mu0 = zeros(n_params,1);
sigma0 = 10*rand(n_params) + 2 * eye(n_params);
sigma0 = sigma0'*sigma0;

% sigma0 = diag(sigma0);

dlmwrite('/tmp/ReLe/chol_FIM/test/mean0.txt', mu0, 'delimiter', '\t', 'precision', 12);
dlmwrite('/tmp/ReLe/chol_FIM/test/sigma0.txt', sigma0, 'delimiter', '\t', 'precision', 12);



ar = [cmd, ' ', '/tmp/ReLe/chol_FIM/test/mean0.txt /tmp/ReLe/chol_FIM/test/sigma0.txt'];

disp('----- command output -----');
status = system(ar);
disp('--------------------------');


Fe = dlmread('/tmp/ReLe/chol_FIM/test/Fe.dat');
Fs = dlmread('/tmp/ReLe/chol_FIM/test/Fs.dat')
Feinv = dlmread('/tmp/ReLe/chol_FIM/test/Feinv.dat');
disp('Fe vs Fs (estimated v.s. computed exactly) -- C++');
[diag(Fe),diag(Fs)]
assert(max(abs(diag(Fe)-diag(Fs))) < 0.1);



policy = gaussian_chol_constant(n_params,mu0,chol(sigma0));
% policy = gaussian_diag_constant(n_params,mu0,sigma0);

Fm = policy.fisher();

Fme = 0;
nbs = 3000;
for o = 1:3000
    t = policy.drawAction();
    g = policy.dlogPidtheta(t);
    Fme = Fme + g * g';
end

Fme = Fme/nbs;

disp('Fm vs Fme (estimated v.s. computed exactly) -- MATLAB');
[diag(Fm), diag(Fme)]
%%

%check that the exact versions are equal between matlab and C++
assert(sum(sum(abs(Fe - Fm)))<1e-4);

Fminv = policy.inverseFisher();
assert(sum(sum(abs(Feinv - inv(Fe)))) < 1e-5);
assert(sum(sum(abs(Fminv - inv(Fm)))) < 1e-5);
assert(sum(sum(abs(Fminv - Feinv))) < 1e-5);