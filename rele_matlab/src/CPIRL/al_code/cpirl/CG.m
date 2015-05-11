function [x, fval, result] = CG(A, b, Np)
%% CENTER OF GRAVITY of a polyhedron {x | Ax<=b}
%%

disp('#### CG ####');

% if isunix() == 1
%     
%     tic
%     
%     dlmwrite('At.dat', A);
%     dlmwrite('bt.dat', b);
%     status = system(['./misc/sampling_convex/build/randomwalk At.dat bt.dat ' num2str(Np) ' out.dat']);
%     x_cg = dlmread('out.dat')';
%     delete('At.dat');
%     delete('bt.dat');
%     
%     t_cg = toc;
%     
% else
%     
%     tic
%     [P, status] = cprnd(Np, A, b);
%     x_cg = mean(P);
%     t_cg = toc;
%     
% end

tic;
[x_cg, status] = cprnd_mex(Np, A, b);
t_cg = toc;

if nargout > 1
    fval = 0;
end
if nargout > 2
    
    if status ~= 0 
        result.exitflag = -1;
    else
        result.exitflag = 0;
    end
    
    result.op_time = t_cg;
end

x = x_cg;

disp('#### Ended ####');

return