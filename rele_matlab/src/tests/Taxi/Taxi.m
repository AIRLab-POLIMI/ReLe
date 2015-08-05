%% Taxi data visualizer
addpath('../Statistics');

%clear old data
clear

%clear old figures
close all

%% Choose file
basedir = '/tmp/ReLe/TaxiFuel/HPG/';
trajectoryFile = [basedir 'TaxiFuel.log'];
agentFile = [basedir 'prova.log'];

%% Read data

disp('Reading data trajectories...')
csv = csvread(trajectoryFile);

disp('Organizing data in episodes...')
episodes = readDataset(csv);
clearvars csv

%% Plot J
plotGradient(1, agentFile);
