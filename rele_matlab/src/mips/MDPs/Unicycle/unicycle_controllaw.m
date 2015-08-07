%%% Unicycle Control Law (deterministic)
%%% Params: 3d vector
classdef unicycle_controllaw < policy
    
    methods
        
        function obj = unicycle_controllaw(init_k)
            obj.dim = 3;
            assert(obj.dim == size(init_k,1))
            
            obj.theta = init_k(:);
            obj.dim_explore = 0;
        end
        
        function probability = evaluate(obj, state, action)
            
            ac = obj.drawAction(state);
            
            if abs(ac(1)-action(1)) < 1e-4 &&  abs(ac(2)-action(2)) < 1e-4
                probability = 1;
            else
                probability = 0;
            end
        end
        
        function action = drawAction(obj, state)
            k1 = params(1);
            k2 = params(2);
            k3 = params(3);
            
            rho   = state(1);
            gamma = state(2);
            delta = state(3);
            
            a1 = k1 * rho * cos(gamma);
            a2 = k2 * gamma + k1 * sin(gamma) * cos(gamma) * (gamma + k3 * delta) / gamma;
            action = [a1;a2];
        end
        
        %%% Differential entropy, can be negative
        function S = entropy(obj, state)
            S = nan;
        end
        
        function obj = makeDeterministic(obj)
        end
        
    end
    
end
