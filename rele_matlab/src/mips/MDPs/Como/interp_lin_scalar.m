function y = interp_lin_scalar( X , Y , x )
%
% y = interp_lin_scalar( X , Y , x )
%
%            Y(k+1) + Y(k)
% y = Y(k) + ------------- ( x - X(k) ) 
%            X(k+1) - X(k)
%
% with 'k' such that X(k) <= x < X(k+1)
%
% input :
% X = vector of independent variables - ( n , 1 )
% Y = vector of dependent variables   - ( n , 1 )
% x = scalar                          - ( 1 , 1 )
%
% output :
% y = scalar                          - ( 1 , 1 )
%
% Last update : Francesca 09/12/2008

% -------------
% extreme cases
% -------------
if x <= X( 1 ) ; y = Y( 1 ) ; return ; end
if x >= X(end) ; y = Y(end) ; return ; end

% -------------
% otherwise
% -------------

% Find index 'k' of subinterval [ X(k) , X(k+1) ] s.t. X(k) <= x < X(k+1)
[ ignore , i ] = min( abs( X - x ) ) ;

% If X( i ) = x     then   y = Y( i ) :
if X( i ) == x ; y = Y( i ) ; return ; end

% Else :
% if X( i ) < x     then   k = i  
% if X( i ) > x     then   k = i - 1
k = i - ( X( i ) > x )   ;       
% Line joining points ( X(k) , Y(k) ) and ( X(k+1) , Y(k+1) ) 
Dy = Y( k + 1 ) - Y( k ) ;
Dx = X( k + 1 ) - X( k ) ;
m  = Dy / Dx             ; % slope
% Interpolate :
y = Y( k ) +  m * ( x - X( k ) ) ;