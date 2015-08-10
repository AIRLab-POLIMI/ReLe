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
                mu = mu / norm(mu);
                
                K = abs(mu) > 1e-6;
                active_feat = active_feat(K & not(const_ft));
                
                % force simplex constraint
                simplex = 1;
            end
            
            if length(active_feat) == 1
                x = zeros(n,1);
                x(active_feat) = 1;
                fval = -99;
                return;
            end
            
            % fix input parameters in the objective function
            objfun = @(x) IRLgrad_step_objfun(obj, x, gradAlgType, active_feat);
            
            % 'Algorithm', 'interior-point', ...
            % 'Display', 'off', ...
            if strcmp(gradAlgType,'enac')
                options = optimset('GradObj', 'off', ...
                    'MaxFunEvals', 300 * 5, ...
                    'TolX', 10^-12, ...
                    'TolFun', 10^-12, ...
                    'MaxIter', 300);
            else
                options = optimset('GradObj', 'on', ...
                    'MaxFunEvals', 300 * 5, ...
                    'TolX', 10^-12, ...
                    'TolFun', 10^-12, ...
                    'MaxIter', 300);
            end
            
            % set uniform weights only on the active parameters
            dim = length(active_feat);
            tic;
            if simplex == 1
                % if simplex last elements is linear dependent
                x0 = ones(dim-1,1) / (dim-1);
                % use simplex constraint ( -x<=0, sum x <= 1)
                [x,fval,exitflag,output] = fmincon(objfun, x0, ...
                    [-eye(dim-1);ones(1,dim-1)], [zeros(dim-1,1);1], [], [],[], [], [], options);
            else
                x0 = ones(dim,1) / (dim);
                % do not use any constraints
                [x,fval,exitflag,output] = fmincon(objfun, x0, ...
                    [], [], [], [], [], [], [], options);
            end
            t = toc;
            
            % save result
            obj.weights = x;
            
        end
        
        function [f,g] = IRLgrad_step_objfun(obj, x, gtype, active_feat)
            
            pol     = obj.policy;
            dataset = obj.data;
            df      = obj.gamma;
            
            full_x = zeros(length(obj.weights),1);
            if (length(x) == length(active_feat)-1)
                full_x(active_feat(1:end-1)) = x;
                full_x(active_feat(end)) = 1 - sum(x);
            else
                full_x(active_feat) = x;
            end
            
            rew_fun = @(state,action,nexts) obj.fReward(state,action,nexts,full_x);
            rew_der = @(state,action,nexts) obj.dfReward(state,action,nexts,full_x);
            
            if strcmp(gtype,'r')
                dgraddomega = eREINFORCE_IRL(pol, dataset, df, rew_der);
                dJdtheta    = eREINFORCE_IRL(pol, dataset, df, rew_fun);
            elseif strcmp(gtype,'rb')
                % dJdtheta    = eREINFORCEbase_IRL(policy, data, gamma, fReward);
                % dgraddomega = eREINFORCEbase_IRL(policy, data, gamma, dfReward);
                [dJdtheta, dgraddomega] = eREINFORCEbase_IRL_grad(pol, dataset, df, rew_fun, rew_der);
                % assert(norm(dJdtheta-dJdtheta2,inf) <= 1e-6);
                % assert(max(max(abs(dgraddomega-dgraddomega2))) <= 1e-6);
            elseif strcmp(gtype,'g')
                dgraddomega = GPOMDP_IRL(pol, dataset, df, rew_der);
                dJdtheta    = GPOMDP_IRL(pol, dataset, df, rew_fun);
            elseif strcmp(gtype,'gb')
                %     dJdtheta    = GPOMDPbase_IRL(policy, data, gamma, fReward);
                %     dgraddomega = GPOMDP_IRL(policy, data, gamma, dfReward);
                %     [dJdtheta, dgraddomega] = GPOMDPbase_IRL_grad(policy, data, gamma, fReward, dfReward);
                [dJdtheta, dgraddomega] = GPOMDPbase_NEW_IRL_grad(pol, dataset, df, rew_fun, rew_der);
            elseif strcmp(gtype,'enac')
                [dJdtheta, dgraddomega] = eNAC_IRL_grad(pol, dataset, df, rew_fun, rew_der);
            else
                error('Unknown gradient type');
            end
            
            % select only active elements in the derivative of the policy
            % gradient
            if (length(x) == length(active_feat)-1)
                dgraddomega = dgraddomega(:,active_feat(1:end-1));
            else
                dgraddomega = dgraddomega(:,active_feat);
            end
            
            % compute score function and its gradient
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
            constant_reward = zeros(n,nbEpisodes);
            
            % scan data
            for ep = 1:nbEpisodes
                nbSteps = size(obj.data(ep).s,2);
                % store immediate reward over trajectory
                reward_vec = zeros(n,nbSteps);
                
                df = 1;
                for t = 1:nbSteps
                    state  = obj.data(ep).s(:,t);
                    action = obj.data(ep).a(:,t);
                    nexts  = obj.data(ep).nexts(:,t);
                    
                    reward_vec(:,t) = obj.dfReward(state, action, nexts, x);
                    mu = mu + df * reward_vec(:,t);
                    
                    df = df * obj.gamma;
                end
                
                % check reward range over trajectories
                R = range(reward_vec,2);
                constant_reward(R<1e-4, ep) = 1;
            end
            
            const_ft = zeros(n,1);
            const_ft( sum(constant_reward,2) == nbEpisodes ) = 1;
            
            mu = mu / nbEpisodes;
        end
        
        %         function mu = compute_features(obj)
        %             n = length(obj.weights);
        %             mu = zeros(n,1);
        %             nbEpisodes = length(ds);
        %             x = zeros(n,1);
        %
        %             % scan data
        %             for ep = 1:nbEpisodes
        %                 nbSteps = size(ds(ep).s,2);
        %                 df = 1;
        %                 for t = 1:nbSteps
        %                     state  = obj.data.state(:,t);
        %                     action = obj.data.action(:,t);
        %                     nexts  = obj.data.nexts(:,t);
        %                     for i = 1:n
        %                         x(i) = 1;
        %                         mu(i) = mu(i) + df * obj.fReward(state, action, nexts, x);
        %                         x(i) = 0;
        %                     end
        %                     df = df * obj.gamma;
        %                 end
        %             end
        %         end
        
    end
    
end

