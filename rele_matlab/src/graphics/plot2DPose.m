function plot2DPose( x, y, theta, sX, sY, c)
%PLOT2DPOSE Plot a 2D pose
%   Plot a triangle centered in (x,y) oriented like theta

% compute the triangle
p1 = [sX*cos(theta) + x, sY*sin(theta) + y];
p2 = [sX*cos(theta + 2*pi/3) + x, sY*sin(theta + 2*pi/3) + y];
p3 = [sX*cos(theta - 2*pi/3) + x, sY*sin(theta - 2*pi/3) + y];

X = [p1(1), p2(1), p3(1); ...
     p2(1), p3(1), p1(1)];
Y = [p1(2), p2(2), p3(2); ...
     p2(2), p3(2), p1(2)];

line(X, Y, 'Color', c);

end

