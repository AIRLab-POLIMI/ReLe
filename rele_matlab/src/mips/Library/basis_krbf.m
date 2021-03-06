function Phi = basis_krbf(n_centers, range, state)
% Uniformly distributed Kernel Radial Basis Functions. Centers and 
% bandwidths are automatically computed to guarantee 25% of overlapping and 
% confidence between 95-99%.
%
% Phi = exp( -(state - centers)' * B^-1 * (state - centers) ), 
% where B is a diagonal matrix denoting the bandwiths of the kernels.
%
% Inputs:
%  - n_centers        : number of centers (the same for all dimensions)
%  - range            : N-by-2 matrix with min and max values for the
%                       N-dimensional input state
%  - state (optional) : the state to evaluate
%
% Outputs:
%  - Phi              : if a state is provided as input, the function 
%                       returns the feature vector representing it; 
%                       otherwise it returns the number of features
%
% Example:
% basis_krbf(2, [0,1; 0,1], [0.2, 0.1]')
%     0.7118
%     0.0508
%     0.0211
%     0.0015

persistent centers

n_features = size(range,1);
b = zeros(1, n_features);
c = cell(n_features, 1);

% Compute bandwidths and centers for each dimension
for i = 1 : n_features
    
    b(i) = (range(i,2) - range(i,1))^2 / n_centers^3;
    m = abs(range(i,2) - range(i,1)) / n_centers;
    c{i} = linspace(-m * 0.1 + range(i,1), range(i,2) + m * 0.1, n_centers);

end

% Compute all centers point
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
    
    B = diag(1./b);
    
    for i = 1 : dim_phi
        x = state - centers(:,i);
        Phi(i) = exp(-x' * B * x);
    end
%     Phi = Phi ./ sum(Phi);

end

% %%% Plotting
% idx = 1;
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
% figure; plot(t(:,idx),u,'Linewidth',2)
    
end
