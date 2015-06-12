function loss = eval_loss(front, domain)

[reference_front, weights] = getReferenceFront(domain, 0);

diff_J = (max(reference_front) - min(reference_front));

loss = 0;

for i = 1 : size(weights,1)
    w = weights(i,:)';
    front_w = front * w;
    reference_front_w = reference_front * w;
    loss = loss + (max(reference_front_w) - max(front_w)) / (diff_J * w);
end

loss = loss / size(weights,1);

end