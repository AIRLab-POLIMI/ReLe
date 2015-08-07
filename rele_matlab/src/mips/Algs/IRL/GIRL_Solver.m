classdef GIRL_Solver < handle
    
    properties(GetAccess = 'public', SetAccess = 'private')
        fReward;
        dfReward;
        data;    % dataset
        policy;  % distribution for sampling the episodes
        weights;
        gamma;
    end
    
    methods
        
        %% CLASS CONSTRUCTOR
        function obj = GIRL_Solver(ds, policy, rewardReg, dRewardReg, gamma)
            obj.fReward = rewardReg;
            obj.dfReward = dRewardReg;
            obj.data = ds;
            obj.policy = policy;
            obj.weights = zeros(feval(rewardReg),1);
            obj.gamma = gamma;
        end
        
        %% SETTER
        function obj = setPolicy(obj, policy)
            obj.policy = policy;
        end
        
        function obj = setData(obj, data)
            obj.data = data;
        end
        
        function obj = setGamma(obj, gamma)
            obj.gamma = gamma;
        end
        
        %% CORE
        function [x, fval] = solve(obj, gradAlgType, simplex)
            objfun = @(x) IRLgrad_step_objfun(obj, x, gradAlgType);
            
            % 'Algorithm', 'interior-point', ...
            % 'Display', 'off', ...
            options = optimset('GradObj', 'on', ...
                'MaxFunEvals', 300 * 5, ...
                'TolX', 10^-12, ...
                'TolFun', 10^-12, ...
                'MaxIter', 300);
            x0 = ones(size(obj.weights))/length(obj.weights);
            tic;
            if simplex == 1
                [x,fval,exitflag,output] = fmincon(objfun, x0, ...
                    -eye(2), zeros(2,1), ones(1,2), 1,[], [], [], options);
            else
                [x,fval,exitflag,output] = fmincon(objfun, x0, ...
                    [], [], [], [], [], [], [], options);
            end
            t = toc;
            
            obj.weights = x;
            
        end
        
        function [f,g] = IRLgrad_step_objfun(obj, x, gtype)
            
            policy = obj.policy;
            data   = obj.data;
            gamma  = obj.gamma;
            
            
            fReward  = @(state,action,nexts) obj.fReward(state,action,nexts,x);
            dfReward = @(state,action,nexts) obj.dfReward(state,action,nexts,x);
            
            if strcmp(gtype,'r')
                dJdtheta    = eREINFORCE_IRL(policy, data, gamma, fReward);
                dgraddomega = eREINFORCE_IRL(policy, data, gamma, dfReward);
            elseif strcmp(gtype,'rb')
                % dJdtheta    = eREINFORCEbase_IRL(policy, data, gamma, fReward);
                % dgraddomega = eREINFORCEbase_IRL(policy, data, gamma, dfReward);
                [dJdtheta, dgraddomega] = eREINFORCEbase_IRL_grad(policy, data, gamma, fReward, dfReward);
                % assert(norm(dJdtheta-dJdtheta2,inf) <= 1e-6);
                % assert(max(max(abs(dgraddomega-dgraddomega2))) <= 1e-6);
            elseif strcmp(gtype,'g')
                dJdtheta    = GPOMDP_IRL(policy, data, gamma, fReward);
                dgraddomega = GPOMDP_IRL(policy, data, gamma, dfReward);
            elseif strcmp(gtype,'gb')
                %     dJdtheta    = GPOMDPbase_IRL(policy, data, gamma, fReward);
                %     dgraddomega = GPOMDP_IRL(policy, data, gamma, dfReward);
                [dJdtheta, dgraddomega] = GPOMDPbase_IRL_grad(policy, data, gamma, fReward, dfReward);
            else
                error('Unknown gradient type');
            end
            
            f = 0.5 * norm(dJdtheta,2)^2;
            g = dgraddomega' * dJdtheta;
        end
        
    end
    
end

