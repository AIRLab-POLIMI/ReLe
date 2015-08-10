%%% Univariate normal policy with state dependant standard deviation: N(phi(s)'*k,rho(s)).
%%% Params: mean.
classdef gaussian_statedepstddev_linear < policy
    
    properties(GetAccess = 'public', SetAccess = 'private')
        basis;
        covReg;
    end
    
    methods
        
        function obj = gaussian_statedepstddev_linear(basis, dim, init_k, cov_func)
            assert(isscalar(dim))
            assert(1 == size(init_k,2))

            obj.basis = basis;
            obj.dim = dim;
            obj.theta = init_k;
            obj.dim_explore = 0;
            obj.covReg = cov_func;
        end
        
        function probability = evaluate(obj, state, action)
            phi = feval(obj.basis, state);
            k = obj.theta;
            mu = phi'*k;
            sigma = feval(obj.covReg, state);
            probability = mvnpdf(action, mu, sigma);
        end
        
        function action = drawAction(obj, state)
            phi = feval(obj.basis, state);
            k = obj.theta;
            mu = phi'*k;
            sigma = feval(obj.covReg, state);
            action = mvnrnd(mu,sigma)';
        end
        
        %%% Differential entropy, can be negative
        function S = entropy(obj, state)
            sigma = feval(obj.covReg, state);
            S = 0.5*log( (2*pi*exp(1))^obj.dim * det(sigma) );
        end
        
        %%% Derivative of the logarithm of the policy
        function dlogpdt = dlogPidtheta(obj, state, action)
            if (nargin == 1)
                dlogpdt = size(obj.theta,1);
                return
            end
            phi = feval(obj.basis, state);
            k = obj.theta;
            mu = phi'*k;
            sigma = feval(obj.covReg, state);
            invS = inv(sigma);

            dlogpdt = 0.5 * phi * (invS + invS') * (action-mu);
        end
        
        function obj = update(obj, direction)
            obj.theta = obj.theta + direction;
        end
        
        function obj = makeDeterministic(obj)
        end
        
        function phi = phi(obj, state)
            if (nargin == 1)
                phi = feval(obj.basis);
                return
            end
            phi = feval(obj.basis, state);
        end
        
        function params = getParams(obj)
            params.theta = obj.theta;
        end
        
    end
    
end
