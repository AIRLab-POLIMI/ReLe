function P = calcP( A, B, Q, R, K, g )

I = eye(size(K,1));

P = (Q + K * R * K) / (I - g * (I + 2 * K + K^2)); % only if A = B = I

%     tolerance = 0.0001;
%     P = I;
%     P2 = Q + g*A'*P*A + g*K'*B'*P*A + g*A'*P*B*K + g*K'*B'*P*B*K + K'*R*K;
%     while ~areAlmostEqual(P,P2,tolerance)
%         P = P2;
%         P2 = Q + g*A'*P*A + g*K'*B'*P*A + g*A'*P*B*K + g*K'*B'*P*B*K + K'*R*K;
%     end

end

