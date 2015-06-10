%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Reference: N Beume, B Naujoks, M Emmerich (2007)
% SMS-EMOA: Multiobjective selection based on dominated hypervolume
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
classdef SMSEMOA_Solver < Genetic_Solver
    
    % S-Metric Selection Evolutionary Multi-Objective Algorithm
    
    properties(GetAccess = 'public', SetAccess = 'private')
        fitness;   % fitness function
    end
    
    methods
        
        %% CLASS CONSTRUCTOR
        function obj = SMSEMOA_Solver(elitism, mutation, fitness, crossover, mutate, max_size)
            obj.elitism = elitism;
            obj.mutation = mutation;
            obj.fitness = fitness;
            obj.crossover = crossover;
            obj.mutate = mutate;
            obj.max_size = max_size;
        end
        
        function values = getFitness ( obj, J )
        % In SMS-EMOA, a solution is ranked higher if removing it from the 
        % population the fitness decreases.
            population_size = size(J,1);
            values = zeros(population_size,1);
            
            for i = 1 : population_size
                front_tmp = J;
                front_tmp(i,:) = []; % Remove the i-th element from the pool
                front_tmp = pareto(front_tmp); % Get the Pareto front
                values(i) = obj.fitness(front_tmp); % Evaluate the fitness
            end
        end
        
    end

end
