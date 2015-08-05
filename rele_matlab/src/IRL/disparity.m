%% Testing the disparity measure

clear
clc

gamma = 0.9;
T = 10;

w = [0.2, 0.7];

R = ones(T, 1)*1000;

Jw = 0;
sGamma = 0;
for t=0:T-1
   Jw = Jw + gamma^t*R(t+1);    
   sGamma = sGamma + gamma^t;  
end

Rm = sum(R)/T;
Rhat = Jw/sGamma;

Dbar = 0;
D = 0;

for t=0:T-1
    Dbar = Dbar + (R(t+1) - Rm)^2;
    D = D + gamma^(2*t)*(R(t+1) - Rhat)^2;
end

