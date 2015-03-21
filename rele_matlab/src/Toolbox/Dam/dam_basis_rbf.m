function phi = dam_basis_rbf(state)

%  state_spec = [0 160];
%  step = 10;
%  nr_centers = diff(state_spec)/step + 1;
%  if nargin == 0
%      phi = nr_centers;
%  else
%      centers = [];
%      for i = state_spec(1) : step : state_spec(2)
%          centers = [centers; i];
%      end
%      widths = ones(nr_centers,1) * 3 * step;
%      phi = zeros((size(centers, 1) + 1), 1);
%
%      index = 1;
%      phi(index) = 1;
%      for i = 1 : size(centers, 1)
%          phi(index + i) = exp(-norm(state - centers(i)) / widths(i));
%      end
%  end


nr_centers = 4;
centers = [0; 50; 120; 160];
widths = [50; 20; 40; 50];

if nargin == 0
    phi = nr_centers + 1;
else
    phi = zeros((size(centers, 1) + 1), 1);
    
    index = 1;
    phi(index) = 1;
    for i = 1 : size(centers, 1)
        phi(index + i) = exp(-norm(state - centers(i)) / widths(i));
    end
end

return;
