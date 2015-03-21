%%% Gaussian with linear mean and constant covariance.
%%% Mean is learned, variance is fixed.
classdef gaussian_policy
    
    properties(GetAccess = 'public', SetAccess = 'private')
        basis;
        dim;
        dim_variance_params;
    end
    
    properties(GetAccess = 'public', SetAccess = 'public')
        theta;
        sigma;
    end
    
    methods
        
        function obj = gaussian_policy(basis, dim, init_k, init_sigma)
            assert(isscalar(dim))
            assert(feval(basis) == size(init_k,1))
            assert(dim == size(init_k,2))
            assert(size(init_sigma,1) == dim)
            assert(size(init_sigma,2) == dim)

            obj.theta = init_k(:);
            obj.basis = basis;
            obj.dim = dim;
            obj.sigma = init_sigma;
            obj.dim_variance_params = 0;
        end
        
        function probability = evaluate(obj, state, action)
            phi = feval(obj.basis, state);
            k = vec2mat(obj.theta,obj.dim);
            mu = k*phi;
            probability = mvnpdf(action, mu, obj.sigma);
        end
        
        function action = drawAction(obj, state)
            phi = feval(obj.basis,state);
            k = vec2mat(obj.theta,obj.dim);
            mu = k*phi;
            action = mvnrnd(mu,obj.sigma)';
        end
        
        % differential entropy, can be negative
        function H = entropy(obj, state)
            H = .5*log(2*pi*exp(obj.dim))*det(obj.sigma);
        end
        
        % derivative of the logarithm of the policy
        function dlpdt = dlogPidtheta(obj, state, action)
            if (nargin == 1)
                % Return the dimension of the vector theta
                dlpdt = size(obj.theta,1);
                return
            end
            phi = feval(obj.basis, state);
            k = vec2mat(obj.theta,obj.dim);
            dlpdt = (obj.sigma)\(action-k*phi)*phi';
            dlpdt = dlpdt(:);
        end
        
        % hessian matrix of the logarithm of the policy
        function hlpdt = hlogPidtheta(obj, state, action)
            phi = feval(obj.basis, state);
            dlogpdt = inv(obj.sigma);
            kr_1 = kron(phi, dlogpdt);
            kr_2 = kron(phi', eye(length(action)));
            hlpdt = -kr_1*kr_2;
        end
        
        function obj = update(obj, direction)
            obj.theta = obj.theta + direction;
        end
        
        function obj = makeDeterministic(obj)
            obj.sigma = 1e-8 * ones(size(obj.sigma));
        end
        
        function phi = phi(obj, state)
            if (nargin == 1)
                % Return the dimension of the vector of basis functions
                phi = feval(obj.basis);
                return
            end
            phi = feval(obj.basis, state);
        end
        
        function params = getParams(obj)
            k = vec2mat(obj.theta,obj.dim);

            params.A = k;
            params.Sigma = obj.sigma;
        end
        
    end
    
end
