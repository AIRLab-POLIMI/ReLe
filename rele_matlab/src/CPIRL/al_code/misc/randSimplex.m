function x = randSimplex(dim)
        tot = 0;
        x = zeros(dim,1);
        for i = 1:dim
            rnd = rand;
            rnd = -log(rnd);
            x(i) = rnd;
            tot = tot + rnd;
        end

        x = x / tot;
return