function env = puddleworld_environment()

env.xmin = 0;
env.xmax = 1;
env.ymin = 0;
env.ymax = 1;
env.goalX = 1;
env.goalY = 1;
env.step = 0.05;

env.radius = 0.1;
% length x side
env.lx = [0.35, 0.2];
% length y side
env.ly = [0.2, 0.4];

% centers of the circles
env.cCx = [0.1, 0.45, 0.4, 0.4];
env.cCy = [0.75, 0.75, 0.8, 0.4];

% corner at the bottom left of the rectangle
env.cRx = [0.1, 0.3];
env.cRy = [0.65, 0.4];

return
