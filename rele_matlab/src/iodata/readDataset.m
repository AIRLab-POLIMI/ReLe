function [ episodes ] = readDataset(A)
%READDATASET read a dataset and creates an appropriate cell array
%   This function takes in input a CSV dataset, and format it as a cell
%   array containing (x, u, xn, r, absorbing, last) for each episode


ds = A(1,1);
da = A(1,2);
dr = A(1,3);

finalCol = 1;
absorbingCol = 2;
xStart = absorbingCol+1;
xEnd = absorbingCol+ds;
uStart = xEnd+1;
uEnd = xEnd+da;
rStart = uEnd+1;
rEnd = uEnd+dr;

episodeN = sum(A(2:end, finalCol));

s1 = struct('x',0,'u',0,'r',0,'absorbing',0);
episodes = repmat(s1,[episodeN 1]);

episodesIndex = [0; find(A(2:end, finalCol) == 1)];
episodesIndex = episodesIndex + 2;

for ep=1:(size(episodesIndex, 1)-1)
    startEpisode = episodesIndex(ep);
    endEpisode = episodesIndex(ep+1)-1;
    episodes(ep).x = A(startEpisode:endEpisode, xStart:xEnd);
    episodes(ep).u = A(startEpisode:endEpisode, uStart:uEnd);
    episodes(ep).r = A(startEpisode:endEpisode, rStart:rEnd);
    episodes(ep).absorbing = A(startEpisode:endEpisode, absorbingCol);
end

end

