function phi = Como_basis_rbf(state)

% fixed basis function on the basis of Simona's work (file:
% OptB_wtest_1.out, line 35)

nr_centers = 4; % number of basis function
centers = [-0.021203956075065027 0.79309489870466787 0.54763673375072819;
    0.74163506245263489 -0.40084026679194412 0.28913565382574125;
    0.89366728852812161 0.505594663609501 0.58026568777799004;
    0.050831284865747076 0.3269599440135994 0.58820935304635991];
widths = [0.75801909103943477 0.14417924991150993 0.20800837202233544;
    0.39707030113708691 0.69899865546932172  0.77093781672325301;
    0.76680016605341816 0.90168610262496363 0.087000247423601934;
    0.64331287847618912 0.62377316448808562 0.55767729002940725];
constant_bias = [5.7302588108697986e-05;
    0.00044236985216441165;
    1.1236321822863482e-05;
    0.18985729865293305];
original_weight_of_the_rbf = [0.42207807778826772;
    0.31454499239390987;
    0.99999211371205377;
    0.38407585786723919];
original_flooded_days_per_year =  2.875;
original_average_square_deficit = 764.51470190167595;

if nargin == 0
    phi = nr_centers + 1;
else
    phi = zeros((size(centers, 1) + 1), 1);
    
    persistent Como_Inflow

    if isempty(Como_Inflow)
        %%% Load inflow observations
        load larioInflow
        Como_Inflow = larioInflow;
    end
    
    % normalization and reordering of state
    state = state(:)'; % to be sure is a row vector
    state_orig = state;
    t_nat = Como_Inflow.tnat(state_orig(2) - Como_Inflow.time(1,1) + 1);
    state(1,1) = sin(2 * pi * (t_nat - 1) / 364);
    state(1,2) = cos(2 * pi * (t_nat - 1) / 364);
    state(1,3) = state_orig(1);
    
    index = 1;
    phi(index) = 1;
    for i = 1 : size(centers, 1)
        phi(index + i) = exp(- sum( ( state - centers(i, :) ).^2 ./ widths(i, :).^2 ) )...
            + constant_bias(i);
    end
    
end

return;

