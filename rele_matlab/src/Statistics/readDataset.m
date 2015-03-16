function [ episodes ] = readDataset(A)
%READDATASET read a dataset and creates an appropriate cell array
%   This function takes in input a CSV dataset, and format it as a cell
%   array containing (x, u, xn, r, absorbing, last) for each episode

ds = A(1,1);
da = A(1,2);
dr = A(1,3);

xStart = 1;
xEnd = ds;
uStart = ds+1;
uEnd = ds+da;
xnStart = ds+da+1;
xnEnd = ds+da+ds;
rStart = ds+da+ds+1;
rEnd = ds+da+ds+dr;
absorbingCol = ds+da+ds+dr+1;

episodeN = sum(A(2:end, end));

episodes = cell(episodeN, 5);


xC = 1;
uC = 2;
xnC = 3;
rC = 4;
absorbingC = 5;

episodesIndex = [0; find(A(1:end, end) == 1)];
episodesIndex = episodesIndex + 1;

for ep=1:(size(episodesIndex, 1)-1)
    startEpisode = episodesIndex(ep);
    endEpisode = episodesIndex(ep+1)-1;
    episodes{ep, xC} = A(startEpisode:endEpisode, xStart:xEnd);
    episodes{ep, uC} = A(startEpisode:endEpisode, uStart:uEnd);
    episodes{ep, xnC} = A(startEpisode:endEpisode, xnStart:xnEnd);
    episodes{ep, rC} = A(startEpisode:endEpisode, rStart:rEnd);
    episodes{ep, absorbingC} = A(startEpisode:endEpisode, absorbingCol);
end

end

