function dam_draw_policies(policies, episodic)
% Plots the policies of the Pareto-frontier. Set episodic to 1 if the
% policies are obtained with an episodic-based approach.

n_pol = length(policies);
[~, policy] = settings('dam');
policy = policy.makeDeterministic;

x = [];
y = [];
z = [];

for h = 1 : n_pol
    
    a = [];
    s = [];
    
    if episodic
        pol_high = policies(h).makeDeterministic;
        dim_theta = pol_high.dim;
        policy.theta(1:dim_theta) = pol_high.drawAction;
    else
        policy = policies(h).makeDeterministic;
    end
    
    for i = 0 : 5 : 160
        a = [a; policy.drawAction(i)];
        s = [s; i];
    end
    x = [x s];
    y = [y a];
    z = [z h*ones(33,1)];

end

figure
for i = 1 : n_pol
    plot3(z(:,i),x(:,i),y(:,i)); hold on;
end

