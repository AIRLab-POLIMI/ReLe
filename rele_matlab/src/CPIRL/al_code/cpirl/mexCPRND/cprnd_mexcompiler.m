clc;
if isunix() 
mex_cmd = ['mex -g -v '...
    'CXXFLAGS=''-fPIC'' '...
    '-largeArrayDims ' ...
    '-I/opt/ibm/ILOG/CPLEX_Studio126/cplex/include ' ...
    '-I/opt/ibm/ILOG/CPLEX_Studio126/concert/include ' ...
    '/opt/ibm/ILOG/CPLEX_Studio126/cplex/lib/x86-64_linux/static_pic/libilocplex.a ' ...
    '/opt/ibm/ILOG/CPLEX_Studio126/concert/lib/x86-64_linux/static_pic/libconcert.a ' ...
    '/opt/ibm/ILOG/CPLEX_Studio126/cplex/lib/x86-64_linux/static_pic/libcplex.a ' ...
    ' -DIL_STD cprnd_mex.cpp cprnd_pure.cpp'];
    
else
mex_cmd = ['mex -g -v '...
    'CXXFLAGS=''-fPIC'' '...
    '-largeArrayDims ' ...
%% Qui bisogna aggiungere le librerie di cplex, ilocplex.lib concert.lib
    ' -DIL_STD cprnd_mex.cpp cprnd_pure.cpp'];
end
eval(mex_cmd)

addpath ./test/
