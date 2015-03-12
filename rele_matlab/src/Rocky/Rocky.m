%% Rocky data visualizer

figure(1)
clf(1)

figure(2)
clf(2)

clear

%% Read data

csv = csvread('/home/dave/prova.txt');

x = csv(:, 1:8);
u = csv(:, 9:11);
xn = csv(:, 12:19);
r = csv(:, 20);

traj = [x(:, 1:2); xn(end, 1:2)];
rockytraj = traj + [x(:, 6:7); xn(end, 6:7)];

figure(1)
hold off
plot(traj(:, 1), traj(:, 2), 'b');
hold on
plot(rockytraj(:, 1), rockytraj(:, 2), 'm');
axis equal



for i = 1:10:size(traj, 1)
   endIndex = min(size(traj, 1), i+10);
   figure(2)
   hold off
   plot(traj(i:endIndex, 1), traj(i:endIndex, 2), 'b');
   hold on
   plot(rockytraj(i:endIndex, 1), rockytraj(i:endIndex, 2), 'm');
   axis([-6 7 -1 7])
   waitforbuttonpress
end

