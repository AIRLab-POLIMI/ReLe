function [x, fval, result] = MVIE(A, b)
%% MAXIMUM VOLUME INSCRIBED ELLIPSOID of a polyhedron {x | Ax<=b}
%%

disp('#### MVIE ####');

% find the largest inner ellipsoidal approximation. Here, the ellipsoid is
% given by { Ex+c | ||x||_2 \le 1 }
% (see Boyd, Vandenberge: Convex Optimization, 2004, section 8.4.2)
dim = size(A,2);

tic;
% % YALMIP
% E = sdpvar(dim, dim);
% c = sdpvar(dim, 1);
% con = [];
% for i = 1:size(A, 1)
%     con = con + [ norm(E*A(i, :)') + A(i, :)*c <= b(i) ];
% end
% con = con + [ E>=0 ];
% solvesdp(con, -logdet(E));


% formulate and solve the problem
cvx_begin
    variable B(dim,dim) symmetric
    variable d(dim)
    maximize( det_rootn( B ) )
    subject to
       for i = 1:size(A,1)
           norm( B*A(i,:)', 2 ) + A(i,:)*d <= b(i);
       end
cvx_end
t_mvie = toc;

% Ein = double(E);
% cin = double(c);
Ein = B;
cin = d;


if nargout > 1
%     fval = -log(det(Ein));
    fval = cvx_optval;
end
if nargout > 2
    result.Ein = Ein;
    result.Cin = cin;
    result.op_time = t_mvie;
    if strcmp(cvx_status, 'Infeasible') == 1
        result.exitflag = -1;
    else
        result.exitflag = 0;
    end
end

x = cin;

disp('#### Ended ####');

return