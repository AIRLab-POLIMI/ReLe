function MCDraw(episodes)
close all;
figure(1);
x = -1.3:0.05:1;
h = Hill(x);
% plot(x,h);
% hold on;

for i = 1:length(episodes)
    for t = 1:size(episodes(i).x,1)
        s  = episodes(i).x(t,2);
        hs = Hill(s);
        
        plot(x,h);
        title(['Episode ', num2str(i)]);
        line([0.5 0.5],[-0.3 0.5], 'Color', [.8 .8 .8]);
        ylim([-0.3, 0.5]);
        hold on;
        plot(s,hs,'ob', 'MarkerFace','b','MarkerSize', 10);
        hold off;
        
        pause(0.01);
    end
end
end