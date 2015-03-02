
dt = 0.01;

theta = 0;
x = 0;
y = 0;

v = 10;
omega = pi;

steps = 10000;
traj = zeros(steps, 3);
for i=1:steps
 thetaM = (2*theta + omega*dt)/2;
 x = x + v * cos(thetaM) * dt;
 y = y + v * sin(thetaM) * dt;
 theta = theta + omega * dt;
 
 traj(i,:) = [x, y, theta];
 
end


figure(1)
plot(traj(:, 1), traj(:, 2));
axis equal

figure(2)
plot(mod(traj(:, 3), 2*pi));