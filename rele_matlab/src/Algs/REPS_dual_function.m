function [g, gd] = REPS_dual_function(eta, J, epsilon)

% Numerical trick
maxJ = max(J);
J = J - maxJ;

N = length(J);

A = sum(exp(J / eta)) / N;
B = sum(exp(J / eta) .* J) / N;

g = eta * epsilon + eta * log(A) + maxJ; % dual function
gd = epsilon + log(A) - B / (eta * A);   % gradient

end