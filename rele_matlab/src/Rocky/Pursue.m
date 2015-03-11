
%% Pursuer test
 clear
 figure(1)
 clf(1)
 figure(2)
 clf(2)

%% now start the test


dt = 0.01;

theta = 0;
x = 0;
y = 0;

xr = 5;
yr = 5;
thetar = 0;

%data for the controller
thetaMold = 0;
vold = 0;

v = 10;
vrmax = 10;
omegarmax = pi;


omega = pi;

steps = 10000;
traj = zeros(steps, 3);
rockytraj = zeros(steps, 3);
deltatraj = zeros(steps, 1);

%while(true)
for i=1:steps    
 %% Pursuer dynamics
 
 %Step 1: pursuer prevision
 xhat = x + vold * cos(thetaMold) * dt;
 yhat = y + vold * sin(thetaMold) * dt;
 thetaDirhat = wrapToPi(atan2(yhat - (y + yr), xhat - (x + xr)));
 
 %Step 2: compute the inputs
 deltaTheta = wrapToPi(thetaDirhat - thetar)
 
 omegarOpt = deltaTheta/dt;
 omegar = max(-omegarmax, min(omegarmax, omegarOpt))
 
 if(abs(deltaTheta) > pi/2)
     vr = 0
 else if(abs(deltaTheta) > pi/4)
        vr = vrmax/2
     else
        vr = vrmax
     end
 end
 
 %Step 3: update pursuer state
 thetarM = (2*thetar + omegar*dt)/2;
 thetar = wrapToPi(thetar + omegar * dt);
 xrabs = x + xr + vr * cos(thetarM) * dt;
 yrabs = y + yr + vr * sin(thetarM) * dt;
 
%omega = pi*sin(i*pi/512);
%omega = i;
omega = pi;
%omega = log(x^2+y^2 +1 );
    
 %% Evader dinamics
 thetaM = (2*theta + omega*dt)/2;
 x = x + v * cos(thetaM) * dt;
 y = y + v * sin(thetaM) * dt;
 theta = wrapToPi(theta + omega * dt);
 
 %Compute pursuer relative position
 xr = xrabs - x;
 yr = yrabs - y;
 
 traj(i,:) = [x, y, theta];
 rockytraj(i, :) = [xrabs, yrabs, thetar];
 deltatraj(i) = norm([xr, yr]);
 
 %% Pursuer save old values for prediction
 thetaMold = thetaM;
 vold = v;
 
end

figure(2)
hold on
plot(traj(:, 3), 'b');
plot(rockytraj(:, 3), 'm');

figure(3)
plot(deltatraj);

figure(1)
hold off
plot(traj(:, 1), traj(:, 2), 'b');
hold on
plot(rockytraj(:, 1), rockytraj(:, 2), 'm');
axis equal
%axis([-10 10 -10 10])


%waitforbuttonpress
%end