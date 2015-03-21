function b = deep_check_black(x,y)

b = true;
mdp_vars = deep_mdpvariables();

for i = 3 : mdp_vars.state_dim(1)
    for j = 1 : i - 2
        if x == i && y == j
            b = false;
            return
        end
    end
end

if (x == 6 && y == 5) || (x == 6 && y == 6) || (x == 7 && y == 6) || (x == 9 && y == 8)
    b = false;
end

end