function val = lqr_rewardfun(state, action, nextstate, w)
dim = 2;
if nargin == 0
    val = dim;
    return
end
LQR = lqr_environment(dim);
val = 0;
for i = 1:dim
    val = val - (state' * LQR.Q{i} * state + action' * LQR.R{i} * action) * w(i);
end
end