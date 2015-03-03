
%% Pursuer test
 clear
 clf


%% now start the test


dt = 0.01;

theta = 0;
x = 0;
y = 0;

xr = 5;
yr = 5;
thetar = 0;

%data for the controller
xold = 0;
yold = 0;

v = 10;
vrmax = 11;
omegarmax = pi;


omega = pi;

steps = 1000;
traj = zeros(steps, 3);
rockytraj = zeros(steps, 3);
for i=1:steps
    
 %Pursuer dynamics
 
 %Step 1: pursuer prevision
 xhat = x + (x - xold);
 yhat = y + (y - yold);
 thetaDirhat = atan2(yhat - (y + yr), xhat - (x + xr));
 
 %Step 2: compute the inputs
 omegarOpt = mod(thetaDirhat - thetar, 2*pi)/dt;
 omegar = max(-omegarmax, min(omegarmax, omegarOpt));
 
 %Step 3: update pursuer state
 thetarM = (2*thetar + omegar*dt)/2;
 thetar = thetar + omegar * dt;
 xrabs = x + xr + vrmax * cos(thetarM) * dt;
 yrabs = y + yr + vrmax * sin(thetarM) * dt;
 
 % update old values
 xold = x;
 yold = y;
    
 %Evader dinamics
 thetaM = (2*theta + omega*dt)/2;
 x = x + v * cos(thetaM) * dt;
 y = y + v * sin(thetaM) * dt;
 theta = theta + omega * dt;
 
 %Compute pursuer relative position
 xr = xrabs - x;
 yr = yrabs - y;
 
 traj(i,:) = [x, y, theta];
 rockytraj(i, :) = [xrabs, yrabs, thetar];
 
end


figure(1)
hold on
plot(traj(:, 1), traj(:, 2), 'b');
plot(rockytraj(:, 1), rockytraj(:, 2), 'm');
axis equal

figure(2)
hold on
plot(mod(traj(:, 3), 2*pi), 'b');
plot(mod(rockytraj(:, 3), 2*pi), 'm');