function plotCircle(x, y, r)
%PLOTCIRCLE Plot a circle in a figure
%   Plot a circle centered in (x,y) with radius r
ang=0:0.01:2*pi; 
xp=r*cos(ang);
yp=r*sin(ang);
plot(x+xp,y+yp);
end

