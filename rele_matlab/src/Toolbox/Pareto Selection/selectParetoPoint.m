function [possiblefrontier, bestpoint] = selectParetoPoint(frontier,variance, minimum, ratio, heur)

dim = size(frontier,2);
minvalues = min(frontier);
maxvalues = max(frontier);

if(sum(ratio) ~= 1)
    error('Error 1337. Please consult with the magical spagetthi monster.') 
end

normedfrontier = bsxfun(@minus,frontier,minvalues);
maxnorm = max(normedfrontier);
normedfrontier = bsxfun(@rdivide, normedfrontier, maxnorm);
minvalues = min(normedfrontier);
maxvalues = max(normedfrontier);

desiredpoint = (ratio .* (maxvalues - minvalues)) + minvalues;
hold all

closestpoint = dsearchn(normedfrontier,desiredpoint);
current = normedfrontier(closestpoint,:);

if(dim == 2)
    scatter(frontier(closestpoint,1),frontier(closestpoint,2),'d','g');
elseif(dim == 3)
    scatter3(frontier(closestpoint,1),frontier(closestpoint,2),frontier(closestpoint,3),'*','g');
end

distances = dist(normedfrontier,current');
possiblepoints = (distances < variance);
if(dim == 2)
    scatter(frontier(possiblepoints,1),frontier(possiblepoints,2),'r');
elseif(dim == 3)
    scatter3(frontier(possiblepoints,1),frontier(possiblepoints,2),frontier(possiblepoints,3),'r')
end
possiblefrontier = normedfrontier(possiblepoints,:);

[a ,b] = ismember(current, possiblefrontier);
best = b(1);
bestpoint = current;
bestratio = 1;
for i = 1 : length(possiblefrontier) - 1
    diff = possiblefrontier(i,:) - current;
    negative = sum(diff(diff < 0));
    positive = sum(diff(diff > 0));
    
    newratio = positive / abs(negative)
    
    if(any(diff == 0))
        continue;
    end
    
    if(newratio > bestratio * heur)
        bestpoint = possiblefrontier(i,:);
        bestratio = newratio;
        best = i;
    end
end

possiblefrontier = frontier(possiblepoints,:);
if(dim == 2)
    scatter(possiblefrontier(best,1),possiblefrontier(best,2),'s','k');
elseif(dim == 3)
    scatter3(possiblefrontier(best,1),possiblefrontier(best,2),possiblefrontier(best,3),'s','k');
end

end

