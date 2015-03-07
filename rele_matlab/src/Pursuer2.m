%


steps = 1000;
dt = 0.01;


traj = zeros(steps, 2);


x = 0;
y = 0;
vx = 0;
vy = 0;

Dx = 0;
Dy = 0;
Dvx = 0;
Dvy = 0;


for i=1:steps
    ax = cos(x);
    ay = sin(x);
    
    %evader model
    vx = vx + ax*dt;
    vy = vy + ay*dt;
    x = x + vx*dt + 1/2*ax*dt^2;
    y = y + vy*dt + 1/2*ay*dt^2;
    
    traj(i, :) = [x, y];
end


figure(1)
hold off
plot(traj(:, 1), traj(:, 2), 'b');
hold on
%plot(rockytraj(:, 1), rockytraj(:, 2), 'm');