function [ r ] = readCSVLines( fileId, Nlines )
%READCSVLINES Read N lines from a CSV file
%   This function takes as input a file descriptor of a CSV file and  the
%   number of lines and returns the matrix representing that set of lines
r = textscan(fileId, '', Nlines, 'Delimiter',',', 'CollectOutput', true);
end

