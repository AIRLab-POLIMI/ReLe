clear all
domain = 'deep';
robj = 1;
trials = 5;

[N, pol, episodes, steps, gamma, is_avg] = settings(domain);

g0 = [];
g1 = [];
g2 = [];
g3 = [];
g4 = [];
g5 = [];

for i = 1 : trials
    [ds, uJ, dJ] = collect_samples(domain,episodes,steps,pol,is_avg,gamma);
    if gamma == 1
        J = uJ;
    else
        J = dJ;
    end
    
    grad0 = eREINFORCE(pol,ds,gamma,robj);
    g0 = [g0 grad0];

    grad1 = eREINFORCEbase(pol,ds,gamma,robj);
    g1 = [g1 grad1];
    
    grad2 = GPOMDP(pol,ds,gamma,robj);
    g2 = [g2 grad2];

    grad3 = GPOMDPbase(pol,ds,gamma,robj);
    g3 = [g3 grad3];
    
    grad4 = eNAC(pol,ds,gamma,robj);
    g4 = [g4 grad4];
    
    grad5 = eNACbase(pol,ds,gamma,robj);
    g5 = [g5 grad5];
    
end

g0
g1
g2
g3
g4
g5

mean0 = sum(g0')'/trials
deviation0 = zeros(length(mean0),1);
mean1 = sum(g1')'/trials
deviation1 = zeros(length(mean1),1);
mean2 = sum(g2')'/trials
deviation2 = zeros(length(mean2),1);
mean3 = sum(g3')'/trials
deviation3 = zeros(length(mean3),1);
mean4 = sum(g4')'/trials
deviation4 = zeros(length(mean4),1);
mean5 = sum(g5')'/trials
deviation5 = zeros(length(mean5),1);
for i = 1 : trials
    deviation0 = deviation0 + abs(g0(:,i)-mean0);
    deviation1 = deviation1 + abs(g1(:,i)-mean1);
    deviation2 = deviation2 + abs(g2(:,i)-mean2);
    deviation3 = deviation3 + abs(g3(:,i)-mean3);
    deviation4 = deviation4 + abs(g4(:,i)-mean4);
    deviation5 = deviation5 + abs(g5(:,i)-mean5);
end
deviation0
deviation1
deviation2
deviation3
deviation4
deviation5
mod0 = norm(deviation0)
mod1 = norm(deviation1)
mod2 = norm(deviation2)
mod3 = norm(deviation3)
mod4 = norm(deviation4)
mod5 = norm(deviation5)
