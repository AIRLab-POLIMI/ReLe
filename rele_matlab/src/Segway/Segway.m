%% Rocky data visualizer
addpath('../Statistics');

%clear old data
clear

%clear old figures
figure(1)
clf(1)

% figure(2)
% clf(2)
% 
% figure(3)
% clf(3)

%% Read data

disp('Reading data trajectories...')
csv = csvread('/tmp/ReLe/segway/BBO/seqway_final.log');

disp('Organizing data in episodes...')
episodes = readDataset(csv);
clearvars csv

%% Display Data
close all
disp('Plotting trajectories...')

l = 1.2; %m
r = 0.2; %m

%%
for i=1:size(episodes, 1)
    x = episodes(i).x;
    
    for step = 1:length(x)
        
        theta = x(step,1);
        
        x_c = 0; y_c = 0.2;
        th = 0:pi/50:2*pi;
        xunit = r * cos(th) + x_c;
        yunit = r * sin(th) + y_c;
        
        x_h = l * sin(theta);
        y_h = l * cos(theta);
        
        figure(1);
        plot(xunit, yunit);
        hold on;
        plot([x_c;x_h],[y_c;y_h]);
        axis equal;
        
        hold off;
    pause(0.01)
    end
end

%% ODE45
x0 = [0.08 0 0];
[t,y] = ode45(@segway_ode, [0, 0.03], x0);
plot (t,y(:,1))




