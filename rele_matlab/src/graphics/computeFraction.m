function [ sX, sY ] = computeFraction( lim, f)
%COMPUTESCALE computes figure fraction
%   given axis limit lim, computes the fraction f of the axis

dy = lim(2) - lim(1);
dx = lim(4) - lim(3);

sX = dx/f;
sY = dy/f;

end

