%%% Gaussian with linear mean and constant covariance.
%%% Mean is learned, variance is fixed.
classdef gaussian_policy
    
    properties(GetAccess = 'public', SetAccess = 'private')
        % These properties are public by default
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
            % Class constructor
            obj.theta = init_k(:);
            obj.basis = basis;
            obj.dim = dim;
            obj.sigma = init_sigma;
            obj.dim_variance_params = 0;
        end
        
        function probability = evaluate(obj, state, action)
            assert(size(state,2) == 1);
            assert(size(action,2) == 1);
            phi = feval(obj.basis, state);
            k = vec2mat(obj.theta,obj.dim);
            mu = k*phi;
            probability = mvnpdf(action, mu, obj.sigma);
        end
        
        function action = drawAction(obj, state)
            assert(size(state,2) == 1);
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
            assert(size(state,2) == 1);
            assert(size(action,2) == 1);
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
        
    end
    
end
