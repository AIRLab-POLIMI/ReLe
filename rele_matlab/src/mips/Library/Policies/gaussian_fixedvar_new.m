%%% Gaussian with linear mean and fixed covariance: N(K*phi,S).
%%% Params: mean.
classdef gaussian_fixedvar_new < policy
    
    properties(GetAccess = 'public', SetAccess = 'private')
        basis;
    end
    
    properties(GetAccess = 'public', SetAccess = 'public')
        sigma;
    end
    
    methods
        
        function obj = gaussian_fixedvar_new(basis, dim, init_k, init_sigma)
            assert(isscalar(dim))
            assert(1 == size(init_k,2))
            assert(dim == size(init_k,1))
            assert(size(init_sigma,1) == dim)
            assert(size(init_sigma,2) == dim)

            obj.theta = init_k;
            obj.basis = basis;
            obj.dim = dim;
            obj.sigma = init_sigma;
            obj.dim_explore = 0;
        end
        
        function probability = evaluate(obj, state, action)
            phi = feval(obj.basis, state);
            k = obj.theta;
            mu = phi'*k;
            probability = mvnpdf(action, mu, obj.sigma);
        end
        
        function action = drawAction(obj, state)
            phi = feval(obj.basis,state);
            k = obj.theta;
            mu = phi'*k;
            action = mvnrnd(mu,obj.sigma)';
        end
        
        %%% Differential entropy, can be negative
        function S = entropy(obj, state)
            S = .5*log(2*pi*exp(obj.dim))*det(obj.sigma);
        end
        
        %%% Derivative of the logarithm of the policy
        function dlpdt = dlogPidtheta(obj, state, action)
            if (nargin == 1)
                dlpdt = size(obj.theta,1);
                return
            end
            phi = feval(obj.basis,state);
            invS = inv(obj.sigma);
            k = obj.theta;
            mu = phi'*k;
            dlpdt = 0.5 * phi * (invS + invS') * (action-mu);
        end
  
        function obj = update(obj, direction)
            obj.theta = obj.theta + direction;
        end
        
        function obj = makeDeterministic(obj)
            obj.sigma = 1e-8 * ones(size(obj.sigma));
        end
        
        function phi = phi(obj, state)
            if (nargin == 1)
                phi = feval(obj.basis);
                return
            end
            phi = feval(obj.basis, state);
        end
        
        function params = getParams(obj)
            k = obj.theta;

            params.A = k;
            params.Sigma = obj.sigma;
        end
        
    end
    
end
