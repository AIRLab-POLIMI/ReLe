function plotHierarchicalGradient( fig, file )
%PLOTHIERARCHICALGRADIENT Read cvs hierarchical gradient file and plot data in figure fig
%   Reads a gradient dataset and plot the J inside the specified figure
figure(fig)
clf(fig)

csv = csvread(file);

index = 1;
ep = 1;

while(index < size(csv, 1))
    [data(ep), index, ~] = ReadHierarchicalGradientStatistics(csv, index);
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


end

