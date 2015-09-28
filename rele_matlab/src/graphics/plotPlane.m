function plotPlane(p, n, tgv, sgv)
%PLOTPLANE plot a plane
%   Given a normal vector and a point, plots a plane. 

[T,S] = meshgrid(tgv, sgv);

b = eye(3);

for i=1:3
    tmp = cross(n, b(:, i));  
    if(any(tmp))
        break;
    end
end

v1 = cross(tmp, n);
v1 = v1/norm(v1);
v2 = cross(v1, n);
v2 = v2/norm(v2);

X = p(1) + T*v1(1) + S*v2(1); 
Y = p(2) + T*v1(2) + S*v2(2);
Z = p(3) + T*v1(3) + S*v2(3);


surf(X,Y,Z, 'FaceColor','None')

end

