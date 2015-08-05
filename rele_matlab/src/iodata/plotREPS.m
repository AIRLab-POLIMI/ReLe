function plotREPS( fig, file )
%PLOTREPS Read cvs gradient file and plot data in figure fig
%   Reads a REPS dataset and plot the J inside the specified figure
figure(fig)
clf(fig)


%% Read data
csv = csvread(file);

disp('Organizing data...')

index = 1;
ep = 1;

while(index < size(csv, 1))
    [data(ep), index] = ReadREPSStatistics(csv, index); 
    ep = ep + 1;
end

clearvars csv

J = [];
for i=1:size(data,2)
    for j=1:size(data(i).policies, 2)
        J = [J; data(i).policies(j).J];
    end
end

plot(J);
axis tight

end

