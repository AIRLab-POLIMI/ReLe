function frames = moviefront(fig, frames, domain, points)
% Plots and records 

figure(fig)
clf

if size(points,2) == 2
    plot(points(:,1),points(:,2),'r+')
elseif size(points,2) == 3
    plot3(points(:,1),points(:,2),points(:,3),'r+')
else
    error('Unable to print in 4+ dimensions.')
end

getReferenceFront(domain,1);
drawnow
frames(numel(frames)+1) = getframe(fig);
