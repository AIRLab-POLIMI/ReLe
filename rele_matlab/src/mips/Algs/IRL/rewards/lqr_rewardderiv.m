function val = lqr_rewardderiv(state, action, nextstate, w)
dim = 2;
LQR = lqr_environment(dim);
val = ones(1,dim);
for i = 1:dim
    val(i) =  - (state' * LQR.Q{i} * state + action' * LQR.R{i} * action);
end
end