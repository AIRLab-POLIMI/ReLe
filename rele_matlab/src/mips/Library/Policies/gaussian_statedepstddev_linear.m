%%% Univariate normal policy with state dependant standard deviation: N(K*phi(s),rho(s)).
%%% Params: mean.
classdef gaussian_statedepstddev_linear < policy
    
    properties(GetAccess = 'public', SetAccess = 'private')
        basis;
        covReg;
    end
    
    methods
        
        function obj = gaussian_statedepstddev_linear(basis, dim, init_k, cov_func)
            assert(isscalar(dim))
            assert(feval(basis) == size(init_k,2))
            assert(dim == size(init_k,1))
            assert(size(init_sigma,1) == dim)
            assert(size(init_sigma,2) == dim)

            obj.basis = basis;
            obj.dim = dim;
            obj.theta = init_k(:);
            obj.dim_explore = 0;
            obj.covReg = cov_func;
        end
        
        function probability = evaluate(obj, state, action)
            phi = feval(obj.basis, state);
            n_k = obj.dim*feval(obj.basis);
            k = vec2mat(obj.theta(1:n_k),obj.dim);
            mu = k*phi;
            sigma = feval(obj.covReg, state);
            probability = mvnpdf(action, mu, sigma);
        end
        
        function action = drawAction(obj, state)
            phi = feval(obj.basis,state);
            n_k = obj.dim*feval(obj.basis);
            k = vec2mat(obj.theta(1:n_k),obj.dim);
            mu = k*phi;
            sigma = feval(obj.covReg, state);
            action = mvnrnd(mu,sigma)';
        end
        
        %%% Differential entropy, can be negative
        function S = entropy(obj, state)
            n_k = obj.dim*feval(obj.basis);
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
            n_k = obj.dim*feval(obj.basis);
            k = vec2mat(obj.theta(1:n_k),obj.dim);
            mu = k*phi;
            sigma = feval(obj.covReg, state);

            dlogpdt_k = sigma \ (action - k * phi) * phi';

            dlogpdt = [dlogpdt_k(:)];
        end
        
        function obj = update(obj, direction)
            obj.theta = obj.theta + direction;
            n_k = obj.dim*feval(obj.basis);
            sigma = feval(obj.covReg, state);
            obj.theta(n_k+1:end) = nearestSPD(sigma);
        end
        
        function obj = makeDeterministic(obj)
            n_k = obj.dim*feval(obj.basis);
            obj.theta(n_k+1:end) = 1e-8;
        end
        
        function phi = phi(obj, state)
            if (nargin == 1)
                phi = feval(obj.basis);
                return
            end
            phi = feval(obj.basis, state);
        end
        
        function params = getParams(obj)
            n_k = obj.dim*feval(obj.basis);
            k = vec2mat(obj.theta(1:n_k),obj.dim);

            params.A = k;
        end
        
        function obj = randomize(obj, factor)
            n_k = obj.dim*feval(obj.basis);
            obj.theta(n_k+1:end) = obj.theta(n_k+1:end) .* factor;
        end
        
    end
    
end
