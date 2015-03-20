clear all;
close all;

data = dlmread('NLS/NLS_OptParamSpace.dat');
Theta = data(:,1:2);
J = data(:,3);

plot3(Theta(:,1),Theta(:,2), J, 'o');

[optJ, idx] = max(J)

optParams = Theta(idx,:)

[~, I] = sort(data(:,3));
sortedData = data(I,:);
sortedData(end-10:end,:)
