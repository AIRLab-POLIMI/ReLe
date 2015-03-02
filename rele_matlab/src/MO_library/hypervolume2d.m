function hv = hypervolume2d(f,r,max_obj)

f = sortrows(f,1);
f(:,1) = f(:,1) / max_obj(1);
f(:,2) = f(:,2) / max_obj(2);
b = f(1,1) - r(1);
h = f(1,2) - r(2);
hv = b*h;
for i = 2 : size(f,1)
    b = f(i,1) - f(i-1,1);
    h = f(i,2) - r(2);
    hv = hv + b*h;
end

return
    