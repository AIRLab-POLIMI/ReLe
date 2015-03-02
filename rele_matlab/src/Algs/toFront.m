clear all
domain = 'lqr';

[N, pol, episodes, steps, gamma, avg_rew_setting, max_obj] = settings(domain);

tolerance = 0.05;
lrate = .1;
i = 0;
theta = [];
f = [];
p = {};

while true
    i = i + 1
    g = zeros(0);
    gn = zeros(0);
    [ds, uJ, dJ, H] = collect_samples(domain,episodes,steps,pol,avg_rew_setting,gamma);
    theta = [theta; pol.theta'];
    if gamma == 1
        J = uJ;
    else
        J = dJ;
    end
    x = (J.*max_obj)'
    f = [f; x];
    p = [p; pol];
    M = zeros(0);
    Mn = zeros(0);
    for d = 1 : N
        g = eNAC(pol,ds,gamma,d);
        M = [M g];
        if norm(g) > 0
            Mn = [Mn g/norm(g)];
        else
            Mn = [Mn g];
        end
    end
    lambda = quadprog(M'*M,zeros(N,1),[],[],ones(1,N),1,zeros(1,N),[]);
    lambdan = quadprog(Mn'*Mn,zeros(N,1),[],[],ones(1,N),1,zeros(1,N),[]);
    dir = M*lambda;
    dirn = Mn*lambdan;
    dev = norm(dir);
    devn = norm(dirn);
    H_dev_devn = [H dev devn]
    if devn < tolerance || dev < tolerance
        break;
    end
    pol = pol.update(lrate*dirn);
end
