function dy = unicycle_ode(t, y, k)

rho   = max(y(1), 1e-6);
gamma = wrapToPi(y(2));
delta = wrapToPi(y(3));

v = k(1);
w = k(2);

dy = ones(3,1);
dy(1) = -v * cos(gamma);
dy(2) = sin(gamma) * v / rho - w;
dy(3) = sin(gamma) * v / rho;

end