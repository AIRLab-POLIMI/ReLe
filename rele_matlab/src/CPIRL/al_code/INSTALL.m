%% INSTALL TOOLBOXES
clc;
if (isunix == 1)
    addpath(genpath('/opt/ibm/ILOG/CPLEX_Studio126/cplex/matlab/'));
    disp('[/opt/ibm/ILOG/CPLEX_Studio126/cplex/matlab/*] Added IBM CPLEX ...');
    
    if ~exist('solvers/mpt3/tbxmanager', 'dir')
        disp('Downloading cvx for linux...')
        cmd = 'rm -rf solvers/cvx cvx ';
        status = system(cmd);
        cmd = 'wget http://web.cvxr.com/cvx/cvx-a64.zip';
        status = system(cmd);
        cmd = 'mv cvx-a64.zip solvers';
        status = system(cmd);
        cmd = 'unzip solvers/cvx-a64.zip';
        status = system(cmd);
        cmd = 'mv cvx solvers';
        status = system(cmd);
        disp('Installing cvx for linux...')
        run('solvers/cvx/cvx_setup');
        
        disp('Installing mpt3 for linux...')
        run('solvers/mpt3/install_mpt3.m');
        
        cd misc/mexCPRND/
        cprnd_mexcompiler
        cd ../../
    end
    
else
    error('Currently not supported automatically');
end

run('solvers/cvx/cvx_startup');
addpath('cpirl/');
addpath('misc/');
addpath('misc/mexCPRND/');
addpath(genpath('solvers/mpt3/tbxmanager/'));

