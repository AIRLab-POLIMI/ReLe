function [dir, lambda] = paretoAscentDir(N_obj, jacobian)

options = optimset('Display', 'off',...
                   'Algorithm', 'interior-point-convex');
lambda = quadprog(jacobian'*jacobian, ...
    zeros(N_obj,1), [], [], ones(1,N_obj), 1, zeros(1,N_obj), ...
    [], ones(N_obj,1)/N_obj, options);
dir = jacobian * lambda;

end