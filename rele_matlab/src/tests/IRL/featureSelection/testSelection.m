close all
clear

format shortE

M = [ 0.1,  0.2,    0.3;
      0.02, 0.045,  0.027;
      1e-6, 0.1,    1e-9;
      1e-5, 1e-7,   5e-8];
  
Mpfa = PFA(M);
Mpca = PCA(M);
Mpl = licols(M);

[R,basiccol] = rref(M);
Mrref = M(:,basiccol); 


Mpfa
Mpca
Mpl
Mrref


svd(Mpfa)