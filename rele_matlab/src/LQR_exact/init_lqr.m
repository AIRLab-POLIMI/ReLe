% LQR = INIT_LQR(N) initializes a LQR with N conflictual objectives.
function LQR = init_lqr ( n )

if n == 1
    
    LQR.g = 0.95;
    LQR.A = 1;
    LQR.B = 1;
    LQR.Q = 1;
    LQR.R = 1;
    LQR.E = 1;
    LQR.S = 0;
    LQR.Sigma = 0.01;
    LQR.x0 = 10;
    return
    
end

LQR.e = 0.1;
LQR.g = 0.9;
LQR.A = eye(n);
LQR.B = eye(n);
LQR.E = eye(n);
LQR.S = zeros(n);
%    LQR.Sigma = zeros(n);
LQR.Sigma = eye(n);
LQR.x0 = zeros(0);

for i = 1 : n
    LQR.x0 = [LQR.x0; 10];
end
%   LQR.x0 = randi([-32,32],n,1);

for i = 1 : n
    LQR.Q{i} = eye(n);
    LQR.R{i} = eye(n);
end

for i = 1 : n
    for j = 1 : n
        if i == j
            LQR.Q{i}(j,j) = 1-LQR.e;
            LQR.R{i}(j,j) = LQR.e;
        else
            LQR.Q{i}(j,j) = LQR.e;
            LQR.R{i}(j,j) = 1-LQR.e;
        end
    end
end

end