%% Plott all the deta from LQR test
close all
clear
clc

%% load data
path = '/tmp/ReLe/';

load([path, 'F.txt'] ,'ascii')
load([path, 'Fs.txt'] ,'ascii')
load([path, 'G.txt'] ,'ascii')
load([path, 'T.txt'] ,'ascii')
load([path, 'F.txt'] ,'ascii')
load([path, 'J.txt'] ,'ascii')