%%% Gaussian with linear mean (with offset) and constant covariance.
%%% Both mean and variance are learned.
classdef full_smart_gaussian_policy
    
    properties(GetAccess = 'public', SetAccess = 'private')
        basis;
        dim;
        dim_variance_params;
    end
    
    properties(GetAccess = 'public', SetAccess = 'public')
        theta;
    end
    
    methods
        
        function obj = ...
                full_smart_gaussian_policy(basis, dim, init_mean, init_k, init_sigma)
            assert(isscalar(dim))
            assert(feval(basis) == size(init_k,2))
            assert(dim == size(init_k,1))
            assert(dim == size(init_mean,1))
            assert(size(init_mean,2) == 1)
            assert(size(init_sigma,1) == dim)
            assert(size(init_sigma,2) == dim)

            obj.basis = basis;
            obj.dim = dim;
            obj.theta = [init_mean; init_k(:); init_sigma(:)];
            obj.dim_variance_params = length(init_sigma(:));
        end
        
        function probability = evaluate(obj, state, action)
            phi = feval(obj.basis, state);
            n_k = obj.dim*feval(obj.basis)+obj.dim;
            offset = obj.theta(1:obj.dim);
            k = vec2mat(obj.theta(obj.dim+1:n_k),obj.dim);
            mu = offset + k*phi;
            sigma = vec2mat(obj.theta(n_k+1:end),obj.dim);
            probability = mvnpdf(action, mu, sigma);
        end
        
        function action = drawAction(obj, state)
            phi = feval(obj.basis,state);
            n_k = obj.dim*feval(obj.basis)+obj.dim;
            offset = obj.theta(1:obj.dim);
            k = vec2mat(obj.theta(obj.dim+1:n_k),obj.dim);
            mu = offset + k*phi;
            sigma = vec2mat(obj.theta(n_k+1:end),obj.dim);
            action = mvnrnd(mu,sigma)';
        end
        
        % differential entropy, can be negative
        function H = entropy(obj, state)
            n_k = obj.dim*feval(obj.basis)+obj.dim;
            sigma = vec2mat(obj.theta(n_k+1:end),obj.dim);
            H = 0.5*log( (2*pi*exp(1))^obj.dim * det(sigma) );
        end
        
        % derivative of the logarithm of the policy
        function dlogpdt = dlogPidtheta(obj, state, action)
            if (nargin == 1)
                % Return the dimension of the vector theta
                dlogpdt = size(obj.theta,1);
                return
            end
            phi = feval(obj.basis,state);
            n_k = obj.dim*feval(obj.basis)+obj.dim;
            offset = obj.theta(1:obj.dim);
            k = vec2mat(obj.theta(obj.dim+1:n_k),obj.dim);
            mu = offset + k*phi;
            sigma = vec2mat(obj.theta(n_k+1:end),obj.dim);

            dlogpdt_offset = sigma\offset;
            dlogpdt_k = sigma\(action-k*phi)*phi';

            A = -.5 * inv(sigma);
            B = .5 * sigma \ (action - mu) * (action - mu)' / sigma;
            dlogpdt_sigma = A + B;
            dlogpdt = [dlogpdt_offset; dlogpdt_k(:); dlogpdt_sigma(:)];
        end
        
        function obj = update(obj, direction)
            obj.theta = obj.theta + direction;
            n_k = obj.dim*feval(obj.basis)+obj.dim;
            sigma = vec2mat(obj.theta(n_k+1:end),obj.dim);
            obj.theta(n_k+1:end) = nearestSPD(sigma);
        end
        
        function obj = makeDeterministic(obj)
            n_k = obj.dim*feval(obj.basis)+obj.dim;
            obj.theta(n_k+1:end) = 1e-8;
        end
        
        function phi = phi(obj, state)
            if (nargin == 1)
                % Return the dimension of the vector of basis functions
                phi = feval(obj.basis);
                return
            end
            phi = feval(obj.basis, state);
        end
        
        function obj = weightedMLUpdate(obj, d, Theta, Phi)
            N = size(Theta,1);
            D = diag(d);
            S = [ones(N,1), Phi];
            W = (S' * D * S + 1e-8 * eye(size(S,2))) \ S' * D * Theta;
            a = W(1,:)';
            A = W(2:end,:)';
            Sigma = zeros(obj.dim);
            for k = 1 : N
                mu = a + A * Phi(k,:)';
                Sigma = Sigma + (d(k) * (Theta(k,:)' - mu) * (Theta(k,:)' - mu)');
            end
            Z = (sum(d)^2 - sum(d.^2)) / sum(d);
            Sigma = Sigma / Z;
            Sigma = nearestSPD(Sigma);
            obj.theta = [a; A(:); Sigma(:)];
        end
        
        function params = getParams(obj)
            n_k = obj.dim*feval(obj.basis)+obj.dim;
            offset = obj.theta(1:obj.dim);
            k = vec2mat(obj.theta(obj.dim+1:n_k),obj.dim);
            sigma = vec2mat(obj.theta(n_k+1:end),obj.dim);
            params.a = offset;
            params.A = k;
            params.Sigma = sigma;
        end
        
        function obj = randomize(obj, factor)
            n_k = obj.dim*feval(obj.basis)+obj.dim;
            obj.theta(n_k+1:end) = obj.theta(n_k+1:end) * factor;
        end
        
    end
    
end
