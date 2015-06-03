Hj = [];
Hit = [];
WW = [];

for oo = 1:20
    [Wout, Eout, MMc_cg, MMc_mvie, MMc_ac, MMc_cc, MMm, MMl, MMp] = run_NIPS15Tests(oo);
    
    Je = Eout'*Wout;
    
    J_cg = MMc_cg(end,:)*Wout;
    J_mvie = MMc_mvie(end,:)*Wout;
    J_ac = MMc_ac(end,:)*Wout;
    J_cc = MMc_cc(end,:)*Wout;
    it_cg = size(MMc_cg,1);
    it_mvie = size(MMc_mvie,1);
    it_ac = size(MMc_ac,1);
    it_cc = size(MMc_cc,1);
    
    J_mwal = MMm(end,:)*Wout;
    it_mwal = size(MMm,1);
    
    J_lpal  = MMl(end,:)*Wout;
    
    J_proj = MMp(end,:)*Wout;
    it_proj = size(MMp,1);
    
    fprintf('ORIG:     %f\n',  Je);
    fprintf('CPIRL_CG: %f, it: %d\n', J_cg, it_cg);
    fprintf('CPIRL_MV: %f, it: %d\n', J_mvie, it_mvie);
    fprintf('CPIRL_AC: %f, it: %d\n', J_ac, it_ac);
    fprintf('CPIRL_CC: %f, it: %d\n', J_cc, it_cc);
    fprintf('MWAL:     %f, it: %d\n', J_mwal, it_mwal);
    fprintf('LPAL:     %f\n', J_lpal);
    fprintf('PROJ:     %f, it: %d\n', J_proj, it_proj);
    
    WW = [WW; Wout'];
    Hj = [Hj; Je, J_cg, J_mvie, J_ac, J_cc, J_mwal, J_lpal, J_proj];
    Hit = [Hit; it_cg, it_mvie, it_ac, it_cc, it_mwal, it_proj];
    
end

%%
save prova6.mat
Jee = Hj(:,1);
Hj = (Hj - repmat(Jee, 1, size(Hj,2)))./repmat(Jee, 1, size(Hj,2));
mj = mean(Hj)
sj = std(Hj)/sqrt(size(Hj,1))
mi = mean(Hit)
si = std(Hit)/sqrt(size(Hit,1))

[Nr,Nc] = size(Hj);
clc
    for j = 2:Nc
        fprintf('$%.3f \\pm %.3f$&', 1000*mj(j), 1000*sj(j))
    end
    
    [Nr,Nc] = size(Hit);
clc
    for j = 1:Nc
        fprintf('$%.3f \\pm %.3f$&', mi(j), si(j))
    end