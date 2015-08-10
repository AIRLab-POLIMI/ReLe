function val = nls_rewardderiv_g1(state, action, nextstate, w)
dim = 2;
if nargin == 0
    val = dim;
    return
end
centers{1} = [0;0];
centers{2} = [10;10];
widths{1}  = diag([1;1]);
widths{2}  = diag([1;1]);
val = zeros(1,dim);
for i = 1:dim
    val(i) = exp(-(state-centers{i})' * widths{i} * (state-centers{i}));
end
end