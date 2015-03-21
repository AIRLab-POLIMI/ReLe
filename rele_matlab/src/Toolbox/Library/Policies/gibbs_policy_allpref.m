%%% Gibbs (soft-max) policy with preferences on all action.
%%% The temperature is fixed.
classdef gibbs_policy_allpref
    
    properties(GetAccess = 'public', SetAccess = 'private')
        basis;
        theta;
        action_list;
        dim_variance_params;
    end
    
    properties(GetAccess = 'public', SetAccess = 'public')
        inverse_temperature;
    end
    
    methods
        
        function obj = gibbs_policy_allpref(bfs, theta, action_list)
            % Class constructor
            obj.basis = bfs;
            obj.theta = theta;
            obj.action_list = action_list;
            obj.inverse_temperature = 1;
            obj.dim_variance_params = 0;
        end
        
        function probability = evaluate(obj, state, action)
            assert(size(state,2) == 1);
            assert(size(action,2) == 1);

            IT = obj.inverse_temperature;
            
            % Assert that the action is one of the discrete known actions
            I = find(obj.action_list == action);
            assert(length(I) == 1);
            
            % Evaluate the bfs of the current state and action
            phi = feval(obj.basis, state, action);
            lin_prod = obj.theta'*phi;
            
            % Compute the sum of all the preferences
            sumexp = 0;
            for i = 1 : length(obj.action_list)
                act = obj.action_list(i);
                local_phi = feval(obj.basis, state, act);
                sumexp = sumexp + exp(IT*obj.theta'*local_phi);
            end
            
            % Compute action probability
            probability = exp(IT*lin_prod)/sumexp;
        end
        
        function action = drawAction(obj, state)
            assert(size(state,2) == 1);
            
            IT = obj.inverse_temperature;
            
            nactions = length(obj.action_list);
            prob_list = zeros(nactions, 1);
            
            % Compute the sum of all the preferences
            sumexp = 0;
            
            for i = 1 : nactions
                act = obj.action_list(i);
                loc_phi = feval(obj.basis, state, act);
                prob_list(i) = exp(IT*obj.theta'*loc_phi);
                sumexp = sumexp + exp(IT*obj.theta'*loc_phi);
            end
            prob_list = prob_list / sumexp;
            prob_list(isnan(prob_list)) = 1;
            prob_list(isinf(prob_list)) = 1;
            try 
                AV = mnrnd(1,prob_list);
                aidx = sum(AV.*obj.action_list);
            catch
                aidx = discretesample(prob_list, 1);
            end
            action = obj.action_list(aidx);
        end
        
        function H = entropy(obj, state)
            assert(size(state,2) == 1);
            
            IT = obj.inverse_temperature;
            
            nactions = length(obj.action_list);
            prob_list = zeros(nactions, 1);
            
            sumexp = 0;
            
            for i = 1 : nactions
                act = obj.action_list(i);
                loc_phi = feval(obj.basis, state, act);
                prob_list(i) = exp(IT*obj.theta'*loc_phi);
                sumexp = sumexp + exp(IT*obj.theta'*loc_phi);
            end
            prob_list = prob_list / sumexp;
            H = 0;
            for i = 1 : nactions
                if isinf(prob_list(i)) || isnan(prob_list(i))
                    prob_list(i) = 1;
                end
                H = H + (-prob_list(i)*log2(prob_list(i)));
            end
            H = H / log2(nactions);
        end
        
        function dlpdt = dlogPidtheta(obj, state, action)
            if (nargin == 1)
                % Return the dimension of the vector theta
                dlpdt = size(obj.theta,1);
                return
            end
            assert(size(state,2) == 1);
            assert(size(action,2) == 1);
            
            IT = obj.inverse_temperature;
            
            % Assert that the action is one of the discrete known actions
            I = find(obj.action_list == action);
            assert(length(I) == 1);
            
            % Compute the sum of all the preferences
            sumexp = 0;
            sumpref = 0; % sum of the preferences
            nactions = length(obj.action_list);
            prob_list = zeros(nactions, 1);
            for i = 1 : nactions
                act = obj.action_list(i);
                loc_phi = feval(obj.basis, state, act);
                prob_list(i) = exp(IT*obj.theta'*loc_phi) ;
                sumexp = sumexp + exp(IT*obj.theta'*loc_phi);
                sumpref = sumpref + IT*loc_phi*prob_list(i);
            end
            sumpref = sumpref / sumexp;
            
            loc_phi = feval(obj.basis, state, action);
            dlpdt = IT*loc_phi - sumpref;
        end
        
        function obj = makeDeterministic(obj)
            obj.inverse_temperature = 1e8;
        end
        
        function obj = update(obj, direction)
            obj.theta = obj.theta + direction;
        end
        
        function obj = randomize(obj, factor)
            obj.theta = obj.theta * factor;
        end
        
    end
    
end
