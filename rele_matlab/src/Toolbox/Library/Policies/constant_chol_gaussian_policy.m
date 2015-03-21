%%% Gaussian with constant mean and covariance.
%%% Both mean and variance are learned.
%%% The covariance is represented using the Cholesky decomposition.
%%% Reference: Sun, Efficient Natural Evolution Strategies, 2009
classdef constant_chol_gaussian_policy
    
    properties(GetAccess = 'public', SetAccess = 'private')
        dim;
        dim_variance_params;
    end
    
    properties(GetAccess = 'public', SetAccess = 'public')
        theta;
    end
    
    methods
        
        function obj = constant_chol_gaussian_policy(dim, init_mean, init_cholA)
            assert(isscalar(dim))
            assert(size(init_mean,1) == dim)
            assert(size(init_mean,2) == 1)
            assert(size(init_cholA,1) == dim)
            assert(size(init_cholA,2) == dim)
            assert(istriu(init_cholA))
            
            obj.dim = dim;
            indices = triu(ones(dim));
            init_cholA = init_cholA(indices==1);
            obj.theta = [init_mean; init_cholA(:)];
            obj.dim_variance_params = length(init_cholA(:));
        end
        
        function probability = evaluate(obj, theta)
            mu = obj.theta(1:obj.dim);
            indices = triu(ones(obj.dim));
            cholA = indices;
            cholA(indices == 1) = obj.theta(obj.dim+1:end);
            probability = mvnpdf(theta, mu, cholA'*cholA);
        end
        
        function action = drawAction(obj)
            mu = obj.theta(1:obj.dim);
            indices = triu(ones(obj.dim));
            cholA = indices;
            cholA(indices == 1) = obj.theta(obj.dim+1:end);
            action = mvnrnd(mu,cholA'*cholA)';
        end
        
        % differential entropy, can be negative
        function H = entropy(obj, state)
            indices = triu(ones(obj.dim));
            cholA = indices;
            cholA(indices == 1) = obj.theta(obj.dim+1:end);
            H = 0.5*log( (2*pi*exp(1))^obj.dim * det(cholA'*cholA) );
        end
        
        % derivative of the logarithm of the policy
        function dlogpdt = dlogPidtheta(obj, theta)
            if (nargin == 1)
                % Return the dimension of the vector theta
                dlogpdt = size(obj.theta,1);
                return
            end
            mu = obj.theta(1:obj.dim);
            indices = triu(ones(obj.dim));
            cholA = indices;
            cholA(indices == 1) = obj.theta(obj.dim+1:end);
            invsigma = cholA \ eye(size(cholA)) / cholA';
            
            dlogpdt_k = invsigma * (theta - mu);
            
            R = cholA' \ (theta - mu) * (theta - mu)' * invsigma;
            dlogpdt_sigma = zeros(obj.dim);
            for i = 1 : obj.dim
                for j = i : obj.dim
                    if i == j
                        dlogpdt_sigma(i,j) = R(i,j) - 1 / cholA(i,j);
                    else
                        dlogpdt_sigma(i,j) = R(i,j);
                    end
                end
            end
            
            indices = triu(ones(obj.dim));
            dlogpdt_sigma = dlogpdt_sigma(indices == 1);
            
            dlogpdt = [dlogpdt_k(:); dlogpdt_sigma(:)];
        end
        
        function obj = update(obj, direction)
            obj.theta = obj.theta + direction;
        end
        
        function obj = makeDeterministic(obj)
            obj.theta(obj.dim+1:end) = 1e-8;
        end
        
        function F = fisher(obj)
            indices = triu(ones(obj.dim));
            cholA = indices;
            cholA(indices == 1) = obj.theta(obj.dim+1:end);
            invsigma = cholA \ eye(size(cholA)) / cholA';
            F_blocks = cell(obj.dim,1);
            for k = 1 : obj.dim
%                 index = obj.dim + 1 - k;
%                 tmp = invsigma(end-index+1:end, end-index+1:end);
                tmp = invsigma(k:end, k:end);
                tmp(1,1) = tmp(1,1) + 1 / cholA(k,k)^2;
                F_blocks{k} = tmp;
            end
            F = blkdiag(invsigma, F_blocks{:});
        end
        
        function invF = inverseFisher(obj)
            indices = triu(ones(obj.dim));
            cholA = indices;
            cholA(indices == 1) = obj.theta(obj.dim+1:end);
            sigma = cholA' * cholA;
            invsigma = cholA \ eye(size(cholA)) / cholA';
            invF_blocks = cell(obj.dim,1);
            for k = 1 : obj.dim
                index = obj.dim + 1 - k;
                tmp = invsigma(end-index+1:end, end-index+1:end);
                tmp(1,1) = tmp(1,1) + 1 / cholA(k,k)^2;
                invF_blocks{k} = eye(size(tmp)) / tmp;
            end
            invF = blkdiag(sigma, invF_blocks{:});
        end
        
        function params = getParams(obj)
            mu = obj.theta(1:obj.dim);
            indices = triu(ones(obj.dim));
            cholA = indices;
            cholA(indices == 1) = obj.theta(obj.dim+1:end);
            sigma = cholA'*cholA;

            params.mu = mu;
            params.Sigma = sigma;
        end
        
        function obj = randomize(obj, factor)
            obj.theta(obj.dim+1:end) = obj.theta(obj.dim+1:end) * factor;
        end
        
    end
    
end
