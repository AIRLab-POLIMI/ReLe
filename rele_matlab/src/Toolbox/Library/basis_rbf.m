function Phi = basis_rbf(n_centers, range, state)
% Phi = basis_rbf(n_centers, range, state)
%
% It computes equally distributed radial basis functions with 25% of 
% overlapping and confidence between 95-99%.
%
% Inputs:
%
%  - n_centers        : number of centers (same for all dimensions)
%  - range            : N-by-2 matrix with min and max values for the
%                       N-dimensional input state
%  - state (optional) : the state to evaluate
%
% Output:
%
%  - Phi              : if a state is provided as input, the function 
%                       returns the feature vector representing it; 
%                       otherwise it returns the size of such vector

persistent centers

n_features = size(range,1);
b = zeros(1, n_features);
c = cell(n_features, 1);

% compute bandwidths and centers for each dimension
for i = 1 : n_features
    
    b(i) = (range(i,2) - range(i,1))^2 / n_centers^3;
    m = abs(range(i,2) - range(i,1)) / n_centers;
    c{i} = linspace(-m * 0.1 + range(i,1), range(i,2) + m * 0.1, n_centers);

end

% compute all centers point
if size(centers,1) == 0

    d = cell(1,n_features);
    [d{:}] = ndgrid(c{:});
    centers = cell2mat( cellfun(@(v)v(:), d, 'UniformOutput',false) )';

end
dim_phi = size(centers,2);

if ~exist('state','var')
    
    Phi = dim_phi;
    
else
    
    Phi = zeros(dim_phi,1);
    
    B = 0.5 * diag(1./b);
    
    for i = 1 : dim_phi
        x = state - centers(:,i);
        Phi(i) = exp(-x' * B * x);
    end
    Phi = Phi ./ sum(Phi);
    
end

%%% Plotting
% idx = 6;
% t = zeros(100, n_features);
% for i = 1 : n_features
%     t(:,i)  = linspace(range(i,1), range(i,2), 100);
% end
% 
% u = zeros(100, n_centers);
% for k = 1 : 100
%     for j = 1 : n_centers
%         u(k,j) = exp(-(t(k,idx) - c{idx}(j))^2 / (2*b(idx)));
%     end
%     u(k,:) = u(k,:) / sum(u(k,:));
% end
% 
% close all
% figure; plot(t(:,idx),u,'Linewidth',2)
    
end
