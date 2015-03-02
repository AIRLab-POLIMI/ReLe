function new_population = SMSEMOA_evolve ( current_population, current_solutions, ...
    elitism, mutation, domain )

population_size = numel(current_population);
offspring = current_population(1).empty(population_size,0);
new_population = current_population(1).empty(population_size,0);

% how many elements of the population will be kept according to the elitism
n_of_elites = ceil( population_size * elitism );

for i = 1 : population_size
    
    % get 2 parents
    idx1 = randi(numel(current_population),1);
    idx2 = randi(numel(current_population),1);
    theta1 = current_population(idx1).theta;
    theta2 = current_population(idx2).theta;
    
    % generate a child with random crossover
    new_theta = zeros(size(theta1,1),1);
    for j = 1 : size(theta1,1)
        if rand < 0.5
            new_theta(j) = theta1(j);
        else
            new_theta(j) = theta2(j);
        end
    end
    child = current_population(idx1);
    child.theta = new_theta;
    
    % mutate
    if rand < mutation
        child.theta = mutate(child.theta);
    end
    
    % add the child to the offspring
    offspring(i) = child;
    
end

% keep the elites of the current population
rankings = sort_population(current_population, current_solutions, domain);
for i = 1 : n_of_elites
    new_population(i) = current_population(rankings(i,1));
end

% sort the offspring and take the remaining best ones
offspring_solutions = evaluate_policies(offspring, domain);
rankings = sort_population(offspring, offspring_solutions, domain);
for i = 1 : numel(offspring) - n_of_elites
    new_population(i+n_of_elites) = offspring(rankings(i,1));
end

end


% Sort the current population according 'loss' metric: a solution is 
% ranked higher if removing it from the pool the loss increases.
function sorted_population = sort_population (policies, solutions, domain)

sorted_population = zeros(numel(policies),2);

for i = 1 : numel(policies)
    % remove the i-th element from the pool
    front_tmp = solutions;
    front_tmp(i,:) = [];
    sorted_population(i,1) = i;
    % evaluate the loss of the pool without the i-th element
    sorted_population(i,2) = -eval_loss(pareto(front_tmp),domain);
%     sorted_population(i,2) = hypervolume2d(pareto(front_tmp),ref_point,max_obj);
end

sorted_population = sortrows(sorted_population,2);

end


function theta = mutate(theta)

idx = randi(length(theta),1);

if rand < 0.5 % add or subtract the mean value
    mutation = mean(theta);
    if rand < 0.5
        mutation = - mutation;
    end
    theta(idx) = theta(idx) + mutation;
else
    mutation = 2; % halve or double
    if rand < 0.5
        mutation = 1 / mutation;
    end
    theta(idx) = theta(idx) * mutation;
end

end
        