function plotOffGradient( fig, file )
%PLOTOFFGRADIENT Read cvs off-policy gradient file and plot data in figure fig
%   Reads a off-policy gradient dataset and plot the J inside the specified figure

figure(fig)
clf(fig)

csv = csvread(file);

index = 1;
ep = 1;

while(index < size(csv, 1))
    [data(ep), index] = ReadOffGradientStatistics(csv, index);
    ep = ep + 1;
end

clearvars csv

J = [];
N = [];
for i=1:size(data,2)
    J = [J; mean(data(i).J)];    
    N = [N; norm(data(i).gradient,2)];
end

plot(J);

axis tight


end

