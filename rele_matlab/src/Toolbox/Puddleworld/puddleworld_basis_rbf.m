function phi = puddleworld_basis_rbf(state, action)

nr_actions = 3;

persistent centers widths

if size(centers, 1) == 0  % Should avoid unnecessary computation of RBF centers as it is costly.
    disp('Recomputing RBF centers and widths...');
    step = 0.05;
    r = 0.1;

    x = 0.1;
    y = 0.75;
    ang = pi/2 : step*10 : 3/2*pi; 
    xp = r*cos(ang);
    yp = r*sin(ang);
    centers = [centers; (x+xp)', (y+yp)'];

    x = 0.45;
    y = 0.4;
    ang = pi : step*10 : 2*pi; 
    xp = r*cos(ang);
    yp = r*sin(ang);
    centers = [centers; (x+xp)', (y+yp)'];

    x = 0.45;
    y = 0.8;
    ang = 0 : step*10 : 8.5/10*pi; 
    xp = r*cos(ang);
    yp = r*sin(ang);
    centers = [centers; (x+xp)', (y+yp)'];

    for i = 0.1 : step : 0.35
        centers = [centers; i, 0.65; i, 0.85];
    end

    for i = 0.4 : step : 0.65
        centers = [centers; 0.35, i; 0.55, i];
    end
    for i = 0.65 : step : 0.8
        centers = [centers; 0.55, i];
    end

    centers = [centers; 0.36, 0.85; 0.05, 0.05; 0.95, 0.95];
    nr_centers = size(centers,1);
    
    widths = 0.3 / 15 * ones(nr_centers, 1);
end

nr_centers = size(centers,1);

if nargin == 0
    phi = (nr_centers + 1) * nr_actions;
elseif nargin == 1
    phi = basis_rbf(state, 1, 1, centers, widths);
else
    phi = basis_rbf(state, action, nr_actions, centers, widths);
end

return;
