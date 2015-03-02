domain = 'lqr';

[N, pol, episodes, steps, gamma, avg_rew_setting, max_obj] = settings(domain);

% Genetic settings
bound = 0.01;
population_size = 50;
elitism = 0.1;
mutation = 0.8;

% Initial population
current_population = pol.empty(population_size,0);
min_theta = min(pol.theta) - 1;
max_theta = max(pol.theta) + 1;
for i = 1 : population_size
    new_pol = pol;
    new_pol.theta = min_theta + (max_theta - min_theta) * rand(length(pol.theta),1);
    current_population(i) = new_pol;
end

iter = 0;
while true

    iter = iter + 1;
    
    % get the Pareto frontier of the current population
    current_solutions = evaluate_policies(current_population, domain);
    
    % evaluate the population 
    current_fitness = eval_loss(pareto(current_solutions),domain);
%     current_fitness = hypervolume2d(pareto(front),ref_point,max_obj);
    
    % stopping condition
    if current_fitness < bound
%         break;
    end
    
    % create a new population
    current_population = SMSEMOA_evolve(current_population,current_solutions,elitism,mutation,domain);
    fprintf( 'Iteration %d, Fitness: %f\n', iter, current_fitness );

end
