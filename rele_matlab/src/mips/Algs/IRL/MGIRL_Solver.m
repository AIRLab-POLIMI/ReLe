classdef MGIRL_Solver < handle
    
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
        function obj = MGIRL_Solver(ds, policy, rewardReg, dRewardReg, gamma)
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
        
        %% CORE
        function [x, fval] = solve(obj, gradAlgType, reducedSpace)
            if nargin < 3
                reducedSpace = 1;
            end
            objfun = @(x) IRLgrad_step_objfun(obj, x, gradAlgType);
            
            % 'Algorithm', 'interior-point', ...
            % 'Display', 'off', ...
            options = optimset('GradObj', 'on', ...
                'MaxFunEvals', 300 * 5, ...
                'TolX', 10^-12, ...
                'TolFun', 10^-12, ...
                'MaxIter', 300);
            if reducedSpace
                x0 = ones(length(obj.weights)-1,1)/length(obj.weights);
            else
                x0 = ones(length(obj.weights),1)/length(obj.weights);
            end
            tic;
            [x,fval,exitflag,output] = fmincon(objfun, x0, ...
                [], [], [], [], [], [], [], options);
            t = toc;
            
            obj.weights = x;
            
        end
        
        function [f,g] = IRLgrad_step_objfun(obj, x, gtype)
            
            policy = obj.policy;
            data   = obj.data;
            gamma  = obj.gamma;
            
            n = length(x);
            if (n == length(obj.weights) - 1)
                sumexp = 1 + sum(exp(x));
                dist_x = [exp(x); 1] / sumexp;
            else
                sumexp = sum(exp(x));
                dist_x = exp(x) / sumexp;
            end
            % A NaN can occur if the exp was Inf
            dist_x(isnan(dist_x)) = 1;
            % Ensure that the sum is 1
            dist_x = dist_x / sum(dist_x);
            
            DGx = zeros(n,n);
            for i = 1:n
                for j = 1:n
                    if i == j
                        DGx(i,j) = exp(x(i)) * (sumexp - exp(x(i))) / sumexp^2;
                    else
                        DGx(i,j) = - exp(x(i)) * exp(x(j)) / sumexp^2;
                    end
                end
            end
            
            subview = @(t) t(1:n);
            fReward  = @(state,action,nexts) obj.fReward (state, action, nexts, dist_x);
            dfReward = @(state,action,nexts) subview(obj.dfReward(state, action, nexts, dist_x)) * DGx;
            
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