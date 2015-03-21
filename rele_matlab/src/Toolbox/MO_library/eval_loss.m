function l = eval_loss(approx_f, domain)

[f, w] = plot_reference_fronts(domain);
n = size(f,2);
dJ = zeros(n,1);

for i = 1 : n
    dJ(i) = max(f(:,i)) - min(f(:,i));
end

l = 0;
for i = 1 : size(w,1)
    wi = w(i,:)';
    approx_fw = approx_f * wi;
    fw = f * wi;
    l = l + (max(fw) - max(approx_fw)) / (dJ'*wi);
end

l = l / size(w,1);

end