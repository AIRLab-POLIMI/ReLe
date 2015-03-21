function reward = puddleworld_reward_distance(state)

p1 = [0.1 0.75;
    0.45 0.75];
p2 = [0.45 0.4;
    0.45 0.8];
radius = 0.1;
ratio = 400;

if state(1) > p1(2,1)
    d1 = norm(state' - p1(2,:));
elseif state(1) < p1(1,1)
    d1 = norm(state' - p1(1,:));
else
    d1 = abs(state(2) - p1(1,2));
end

if state(2) > p2(2,2)
    d2 = norm(state' - p2(2,:));
elseif state(2) < p2(1,2)
    d2 = norm(state' - p2(1,:));
else
    d2 = abs(state(1) - p2(1,1));
end

min_distance_to_puddle = min(d1, d2);
if min_distance_to_puddle > radius
    reward = 0;
else
    reward = - ratio * (radius - min_distance_to_puddle);
end

return;