% Plots the policies of the Pareto-frontier.

n_pol = length(front_pol);

x = [];
y = [];
z = [];

for h = 1 : n_pol
    
    a = [];
    s = [];
    
    policy = front_pol(h);
    policy = policy.makeDeterministic;
    for i = 0 : 5 : 160
        a = [a; policy.drawAction(i)];
        s = [s; i];
    end
    x = [x s];
    y = [y a];
    z = [z h*ones(33,1)];

end

for i = 1 : n_pol
    plot3(z(:,i),x(:,i),y(:,i)); hold on;
end
