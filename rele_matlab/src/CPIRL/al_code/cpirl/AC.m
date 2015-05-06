function [x, fval, result] = AC(A, b)
%% ANALYTIC CENTER of a polyhedron {x | Ax<=b}
%%

disp('#### AC ####');

% http://users.isy.liu.se/johanl/yalmip/pmwiki.php?n=Tutorials.GeneralConvexProgramming

n = size(A,2);
% m = size(A,1);

% Analytic center
tic;
% YALMIP
x = sdpvar(n,1);
diagnostics = optimize([],-geomean(b-A*x));

% %CVX
% cvx_begin
%     variable x(n)
%     minimize -sum(log(b-A*x))
% cvx_end
t_ac = toc;

x = double(x);
if nargout > 1
    fval = -sum(log(b-A*x));
end
if nargout > 2
    result.op_time = t_ac;
    if diagnostics.problem ~= 0 
        result.exitflag = -1;
    else
        result.exitflag = 0;
    end
end

disp('#### Ended ####');

return