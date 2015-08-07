function dy = segway_ode(t, y, action)
theta  = y(1);
omegaP = y(2);
omegaR = y(3);

mdp_vars = segway_mdpvariables();
Mp  = mdp_vars.Mp;
Mr  = mdp_vars.Mr;
Ip  = mdp_vars.Ip;
Ir  = mdp_vars.Ir;
l   = mdp_vars.l; %m
r   = mdp_vars.r;
g   = mdp_vars.g;


tau = action;

h1 = (Mr+Mp)*r*r+Ir;
h2 = Mp*r*l*cos(theta);
h3 = l*l*Mp+Ip;

dy = zeros(3,1);
dy(1) = omegaP;
dy(2) = -(h2*l*Mp*r*sin(theta)*(omegaP)^2-g*h1*l*Mp*sin(theta)+(h2+h1)*tau)/(h1*h3-h2^2);
dy(3) =  (h3*l*Mp*r*sin(theta)*(omegaP)^2-g*h2*l*Mp*sin(theta)+(h3+h2)*tau)/(h1*h3-h2^2);

end