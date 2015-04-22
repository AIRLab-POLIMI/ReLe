function dy = segway_ode(t,y)
theta = y(1);
omegaP = y(2);
omegaR = y(3);
tau = 0;

Mp = 10;
Mr = 15;
Ip = 19;
Ir = 19;
l = 1.2; %m
r = 0.2;
g = 9.81;

h1 = (Mr+Mp)*r*r+Ir;
h2 = Mp*r*cos(theta);
h3 = l*l*Mp+Ip;

dy = zeros(3,1);
dy(1) = omegaP;
dy(2) = (h3 * l * Mp * r * sin(theta) ...
    * omegaP^2 - g * h1 * l * Mp * sin(theta) ...
    + (h3 + h1) * tau) / (h3^2 - h1 * h2);
dy(3) = (h2 * l * Mp * r * sin(theta) ...
    * omegaP^2 - g * h3 * l * Mp * sin(theta) ...
    + (h3 + h2) * tau) / (h3^2 - h1 * h2);
end