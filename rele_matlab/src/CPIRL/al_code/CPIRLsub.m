function output = CPIRLsub( THETA, F, GAMMA, T, E, INIT_FLAG, query_function, tol )
%% settings
try
    
    nbRuns             = T;
    mu_expert          = E';
    L2normalization    = 1;
    minimalPolyRep     = 0;
    
    % utopic flags
    muExpUtopia        = mu_expert;
    utopic_expert      = 0;
    infeasible         = 0;
    counter = 0;
    
    [N, K] = size(F);
    
    % Choose initial features expectations randomly
    VV = rand(N, K);
    VV = sparse(VV);
    
    %% initialize output
    dim  = K;
    output.weights   = -99*ones(dim, nbRuns);
    %     output.cdist     = -99*ones(mdpconfig.RewardSize, nbRuns);
    output.featexp   = -99*ones(dim, nbRuns);
    output.per_iter_time = -99*ones(1, nbRuns);
    %     output.mdp_solutions = repmat(struct('sol',[]), nbRuns, 1);
    %     output.minmaxObj = -99*ones(1, nbRuns);
    
    %% initial step
    fprintf('\nUniform weights\n');
    % weights are set uniform
    weights = ones(dim,1) / dim;
    
    %% Initialize polyhedron
    ndim = dim -1;
    
    % non normalizzati
    if L2normalization == 1
        Asimplex = [-eye(ndim);    ones(1,ndim)/norm(ones(1,ndim),2)];
        bsimplex = [zeros(ndim,1); 1/norm(ones(1,ndim),2)];
    else
        Asimplex = [-eye(ndim);    ones(1,ndim)];
        bsimplex = [zeros(ndim,1); 1];
    end
    
    %% iterate
    step = 0;
    mapc = colormap;
    idx  = randi(size(mapc,1), nbRuns+1,1);
    while step <= nbRuns
        
        tic;
        
        if infeasible == 0
            fprintf('\n### Iteration %d\n', step);
            
            %% solve MDP
            fprintf('Solving MDP and computing feature expectation\n');
            [P, M, VV] = opt_policy_and_feat_exp(THETA, F, GAMMA, weights, INIT_FLAG, VV, tol);
            % M: The 1 x K vector of "feature expectations" for the optimal policy.
            mu_agent  = M';
            
            
            %% save history
            output.weights(:, step+1)    = weights;
            output.featexp(:, step+1)    = mu_agent;
            
            %% check terminal condition
            
            %             if ~isempty(params.expertWeights) && (params.checkJ)
            %                 jexp = params.expertWeights' * mu_expert;
            %                 jage = params.expertWeights' * mu_agent;
            %                 jexp
            %                 jage
            %                 perr = params.errorFromExp * abs(jexp);
            %                 distanceJ = norm(jage - jexp) - perr;
            %                 if (distanceJ < 0)
            %                     break;
            %                 end
            %             end
            
            %% Check dominance
            distanceFE = max(mu_expert - mu_agent);
            fprintf('Dominance (< 0 stop): %.6f \n', distanceFE);
            if (distanceFE <= 0)
                break;
            end
        end
        
        %% Compute the new expertFeatExp based on utopia concept
        
        Fm = output.featexp(:, 1:step+1);
        Fm = Fm';
        
        if (utopic_expert == 1)
            Wm = output.weights(:, 1:step+1);
            Wm = Wm';
            utA  = Wm;
            utB = zeros(size(Fm,1),1);
            for i = 1:size(Fm,1)
                utB(i)  = Fm(i,:) * Wm(i,:)';
            end
            utobj = -ones(1,dim);
            utA = [utA; -eye(dim)];
            utB = [utB; -mu_expert];
            [x,fval,exitflag] = cplexlp(utobj,utA, utB);
            muExpUtopia = x;
            infeasible = 0;
            if length(x) == 0
                break;
            end
        end
        
        
        %% Add cutting plane to polyhedron
        
        A = Asimplex;
        b = bsimplex;
        for i = 1: size(Fm,1) %forall the features
            featdiff = Fm(i,:) - muExpUtopia';
            if L2normalization == 1
                featdiff = featdiff / norm(featdiff,2);
            end
            av = featdiff(1:end-1) - repmat(featdiff(end),1,dim-1);
            bv = -featdiff(end);
            A = [A; av];
            b = [b; bv];
        end
        
        if minimalPolyRep == 1
            tic;
            P = Polyhedron('A', A, 'b', b);
            P.minHRep();
            tminrep = toc;
            size(A)
            A = P.A;
            b = P.b;
            size(A)
        end
        
        % normalization (norm[2]{a_i} =1)
        if L2normalization == 1
            Anorm = sqrt(sum(abs(A).^2,2));
            An    = A ./ repmat(Anorm,1,size(A,2));
            bn    = b ./ Anorm;
        else
            An = A;
            bn = b;
        end
        
        %%  Select query point
        if size(An,2) == 1
            tic;
            P = Polyhedron('A', An, 'b', bn);
            P.computeVRep();
            if length(P.V) <= 1
                qp_output.exitflag = -1;
            else
                qp_output.exitflag = 0;
            end
            x = mean(P.V);
            fval = 99;
            t_queryp = toc;
            qp_output
        else
            tic;
            [x, fval, qp_output] = feval(query_function, An, bn);
            t_queryp = toc;
        end
        
        if qp_output.exitflag == -1
            
            if utopic_expert == 1
%                 error('A questo punto');
                    counter = counter + 1;
                    if counter == 10
                        break;
                    end
            else
                counter = 0;
            end
            utopic_expert = 1;
            infeasible    = 1;
            warning('Infeasible Query Problem');
            
        else
            
            weights(1:dim-1) = x;
            
            %             if (ndim <= 3)
            %
            %                 P = Polyhedron('A', An, 'b', bn);
            %                 P.plot('color', mapc(idx(step+1),:));
            %                 hold on;
            %                 if length(weights) == 4
            %                     plot3(weights(1),weights(2),weights(3), 'sg');
            %                     text(weights(1),weights(2),weights(3), num2str(step))
            %                 elseif length(weights) == 3
            %                     plot(weights(1),weights(2), 'sg');
            %                     text(weights(1),weights(2), num2str(step))
            %                 else
            %                     plot(weights(1),0, 'sg');
            %                     text(weights(1),0, num2str(step))
            %                 end
            %             end
            
            weights(dim) = 1 - sum(x);
            
            fprintf('time (query point): %f\n', t_queryp);
            
            
            %%
            
            output.querypointval(:, step+1) = fval;
            
            [distanceW, idxmax]  = max(abs(weights - output.weights(:, step+1)));
            fprintf('W   distance: %.6f (component %d)\n', distanceW, idxmax);
            if distanceW <= 1e-10
                warning('Minmax problem returned the weights of the previous iteration');
                break;
                keyboard;
            end
            
            %% increment iteration number
            step = step + 1;
            
            output.per_iter_time(step) = toc;
            
        end
        
    end
    
    output.per_iter_time(step+1) = toc;
    %     output.weights(:, step+1)    = weights;
    %     fprintf('Solving MDP and computing feature expectation\n');
    %     % compute optimal policy
    %     solution = feval(mdpsolver, mdpconfig, weights);
    %     % simulate trajectories
    %     mu_agent  = feval(mdpcollect_samples, mdpconfig, ...
    %         solution, nbEpisodes, nbSteps);
    %     output.featexp(:, step+1)    = mu_agent;
    %     output.mdp_solutions(step+1) = solution;
    
    
    %% Truncate to last iteration
    output.weights(:,step+2:end)       = [];
    output.per_iter_time(:,step+2:end) = [];
    output.featexp(:,step+2:end)       = [];
    % output.minmaxObj(:,step+1:end)     = [];
    % output.poly.A = Ared;
    % output.poly.b = bred;
    % output.poly.C = Cred;
    % output.poly.d = dred;
    
catch err
    disp(err);
    err.stack
    keyboard;
end
return
