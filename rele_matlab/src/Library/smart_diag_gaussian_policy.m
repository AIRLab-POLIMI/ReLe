%%% Gaussian with linear mean and constant diagonal covariance.
%%% Both mean and variance are learned.
classdef smart_diag_gaussian_policy
    
    properties(GetAccess = 'public', SetAccess = 'private')
        % These properties are public by default
        basis;
        dim;
        dim_variance_params;
    end
    
    properties(GetAccess = 'public', SetAccess = 'public')
        theta;
    end
    
    methods
        
        function obj = smart_diag_gaussian_policy(basis, dim, init_k, init_sigma)
            % Class constructor
            obj.basis = basis;
            obj.dim = dim;
            obj.theta = [init_k(:); init_sigma];
            obj.dim_variance_params = length(init_sigma);
        end
        
        function probability = evaluate(obj, state, action)
            assert(size(state,2) == 1);
            assert(size(action,2) == 1);
            phi = feval(obj.basis, state);
            n_k = obj.dim*feval(obj.basis);
            k = vec2mat(obj.theta(1:n_k),obj.dim);
            mu = k*phi;
            sigma = diag(obj.theta(n_k+1:end));
            probability = mvnpdf(action, mu, sigma.^2);
        end
        
        function action = drawAction(obj, state)
            assert(size(state,2) == 1);
            phi = feval(obj.basis,state);
            n_k = obj.dim*feval(obj.basis);
            k = vec2mat(obj.theta(1:n_k),obj.dim);
            mu = k*phi;
            sigma = diag(obj.theta(n_k+1:end));
            action = mvnrnd(mu,sigma.^2)';
        end
        
        % differential entropy, can be negative
        function H = entropy(obj, state)
            n_k = obj.dim*feval(obj.basis);
            sigma = diag(obj.theta(n_k+1:end));
            H = 0.5*log( (2*pi*exp(1))^obj.dim * det(sigma.^2) );
        end
        
        % derivative of the logarithm of the policy
        function dlogpdt = dlogPidtheta(obj, state, action)
            if (nargin == 1)
                % Return the dimension of the vector theta
                dlogpdt = size(obj.theta,1);
                return
            end
            assert(size(state,2) == 1);
            assert(size(action,2) == 1);
            phi = feval(obj.basis, state);
            n_k = obj.dim*feval(obj.basis);
            k = vec2mat(obj.theta(1:n_k),obj.dim);
            mu = k*phi;
            sigma = obj.theta(n_k+1:end);

            dlogpdt_k = sigma.^-2 .* (action - k * phi) * phi';
            dlogpdt_sigma = -sigma.^-1 + (action - mu).^2 ./ sigma.^3;

            dlogpdt = [dlogpdt_k(:); dlogpdt_sigma];
        end
        
        function obj = update(obj, direction)
            obj.theta = obj.theta + direction;
        end
        
        function obj = makeDeterministic(obj)
            n_k = obj.dim*feval(obj.basis);
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
            Sigma = zeros(obj.dim);
            D = diag(d);
            N = size(Theta,1);
            W = (Phi' * D * Phi + 1e-8 * eye(size(Phi,2))) \ Phi' * D * Theta;
            W = W';
            for k = 1 : N
                Sigma = Sigma + (d(k) * (Theta(k,:)' - W*Phi(k,:)') * (Theta(k,:)' - W*Phi(k,:)')');
            end
            Z = (sum(d)^2 - sum(d.^2)) / sum(d);
            Sigma = Sigma / Z;
            Sigma = diag(diag(Sigma));
            obj.theta = [W(:); diag(sqrt(Sigma))];
        end
        
    end
    
end
