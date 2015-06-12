% clear all;
close all;

% [frontier, weights, utopia, antiutopia] = getReferenceFront('lqr',1);

%After finding the closest point to the ratio, how far around we check?
variance = 0.9;
%Defines the minimal percentage of each objective taken.
minimum = [0.1 0.1];
%Needs to sum up to 1 and expresses the preference on objectives.
ratio = [0.1, 0.2, 0.7];
%Value above 1
heur = 10;


[possiblefrontier, best] = selectParetoPoint(f,variance, minimum, ratio, heur);


view(154,10)

hold all
plot3(y(:,1),y(:,2),y(:,3),'.')
xlabel J_1
ylabel J_2
zlabel J_3
grid on

