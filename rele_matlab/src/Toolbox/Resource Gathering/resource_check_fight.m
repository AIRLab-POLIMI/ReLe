function fight = resource_check_fight(state)

fight = 0;
if (state(1) == 1 && state(2) == 4) || (state(1) == 2 && state(2) == 3)
    r = rand();
    if r < 0.1
        fight = 1;
    else
        fight = 0;
    end
end

end