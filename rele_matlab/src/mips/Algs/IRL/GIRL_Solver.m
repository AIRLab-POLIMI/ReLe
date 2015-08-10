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
        function [x, fval] = solve(obj, gradAlgType, simplex, linearReg)
            if nargin < 3
                simplex = 1;
            end
            if nargin < 4
                linearReg = 0;
            end
            
            % number of reward parameters
            n = length(obj.weights);
            % set all the parameters as active (can be reduced if reward is
            % linear)
            active_feat = 1:n;
            
            if linearReg
                % performs preprocessing in order to remove the features
                % that are constant and the one that are almost never
                % under the given samples
                [mu, const_ft] = preproc_linear_reward(obj);
                K = abs(mu) > 1e-5;
                active_feat = active_feat(K & not(const_ft));
            end
            
            % fix input parameters in the objective function
            objfun = @(x) IRLgrad_step_objfun(obj, x, gradAlgType, active_feat);
            
            % 'Algorithm', 'interior-point', ...
            % 'Display', 'off', ...
            options = optimset('GradObj', 'on', ...
                'MaxFunEvals', 300 * 5, ...
                'TolX', 10^-12, ...
                'TolFun', 10^-12, ...
                'MaxIter', 300);
            
            % set uniform weights only on the active parameters
            x0 = ones(length(active_feat),1) / length(active_feat);
            tic;
            if simplex == 1
                % use simplex constraint
                [x,fval,exitflag,output] = fmincon(objfun, x0, ...
                    -eye(2), zeros(2,1), ones(1,2), 1,[], [], [], options);
            else
                % do not use any constraints
                [x,fval,exitflag,output] = fmincon(objfun, x0, ...
                    [], [], [], [], [], [], [], options);
            end
            t = toc;
            
            % save result
            obj.weights = x;
            
        end
        
        function [f,g] = IRLgrad_step_objfun(obj, x, gtype, active_feat)
            
            policy = obj.policy;
            data   = obj.data;
            gamma  = obj.gamma;
            
            full_x = zeros(length(obj.weights),1);
            full_x(active_feat) = x;
            
            fReward  = @(state,action,nexts) obj.fReward(state,action,nexts,full_x);
            dfReward = @(state,action,nexts) obj.dfReward(state,action,nexts,full_x);
            
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
                dgraddomega = GPOMDP_IRL(policy, data, gamma, dfReward);
                dJdtheta    = GPOMDP_IRL(policy, data, gamma, fReward);
            elseif strcmp(gtype,'gb')
                %     dJdtheta    = GPOMDPbase_IRL(policy, data, gamma, fReward);
                %     dgraddomega = GPOMDP_IRL(policy, data, gamma, dfReward);
                %     [dJdtheta, dgraddomega] = GPOMDPbase_IRL_grad(policy, data, gamma, fReward, dfReward);
                [dJdtheta, dgraddomega] = GPOMDPbase_NEW_IRL_grad(policy, data, gamma, fReward, dfReward);
            else
                error('Unknown gradient type');
            end
            
            f = norm(dJdtheta,2)^2;
            g = 2.0 * dgraddomega' * dJdtheta;
            
            fprintf('g2:  %f\n', dJdtheta'*dJdtheta);
            fprintf('f:   %f\n', f);
            fprintf('dwdj:  ');
            disp(num2str(dgraddomega, '%10.5e '));
            fprintf('df:  ');
            disp(num2str(g', '%10.5e '));
            fprintf('x:   ');
            disp(num2str(x', '%10.5g '));
            fprintf('-----------------------------------------\n');
        end
        
        function [mu, const_ft] = preproc_linear_reward(obj)
            % Return feature expectation and a vector that defines the
            % features that are constant (logic vector): values to 1
            % indicate features that are constant
            n = length(obj.weights);
            mu = zeros(n,1);
            nbEpisodes = length(obj.data);
            x = zeros(n,1);
            constant_reward = ones(n,nbEpisodes);
            
            % scan data
            for ep = 1:nbEpisodes
                nbSteps = size(obj.data(ep).state,2);
                % store immediate reward over trajectory
                reward_vec = zeros(n,nbsteps);
                
                df = 1;
                for t = 1:nbSteps
                    state  = obj.data(ep).state(:,t);
                    action = obj.data(ep).action(:,t);
                    nexts  = obj.data(ep).nexts(:,t);
                    for i = 1:n
                        x(i) = 1;
                        reward_vec(i,t) = obj.fReward(state, action, nexts, x);
                        mu(i) = mu(i) + df * reward_vec(i,t);
                        x(i) = 0;
                    end
                    df = df * obj.gamma;
                end
                
                % check reward range over trajectories
                R = range(reward_vec,2);
                constant_reward(R<1e-4, ep) = 1;
            end
            
            const_ft = zeros(n,1);
            const_ft( sum(constant_reward,2) == nbEpisodes ) = 1;
        end
        
        function mu = compute_features(obj)
            n = length(obj.weights);
            mu = zeros(n,1);
            nbEpisodes = length(ds);
            x = zeros(n,1);
            
            % scan data
            for ep = 1:nbEpisodes
                nbSteps = size(ds(ep).s,2);
                df = 1;
                for t = 1:nbSteps
                    state  = obj.data.state(:,t);
                    action = obj.data.action(:,t);
                    nexts  = obj.data.nexts(:,t);
                    for i = 1:n
                        x(i) = 1;
                        mu(i) = mu(i) + df * obj.fReward(state, action, nexts, x);
                        x(i) = 0;
                    end
                    df = df * obj.gamma;
                end
            end
        end
        
    end
    
end

