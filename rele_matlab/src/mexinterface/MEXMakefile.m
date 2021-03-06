clear all;
homePath = getenv('HOME');
pathToReLeSrc = [homePath, '/Projects/github/ReLe/rele/'];
pathToReLeLib = [homePath, '/Projects/github/ReLe/rele-build/'];
gcc_path = '/usr/local/bin/gcc4.7';
cpp_path = '/usr/local/bin/g++4.7';
mexCmd = ['mex -v -g CXXFLAGS=''-std=c++11 -fPIC'' '...
    ' GCC=''',gcc_path,''' CXX=''',cpp_path,''' ' ...
    ' -DARMA_DONT_USE_CXX11' ...
    ' -I' pathToReLeSrc 'include/rele/core' ...
    ' -I' pathToReLeSrc 'include/rele/policy' ...
    ' -I' pathToReLeSrc 'include/rele/utils' ...
    ' -I' pathToReLeSrc 'include/rele/generators' ...
    ' -I' pathToReLeSrc 'include/rele/algorithms' ...
    ' -I' pathToReLeSrc 'include/rele/environments' ...
    ' -I' pathToReLeSrc 'include/rele/approximators' ...
    ' -I' pathToReLeSrc 'include/rele/statistics' ...
    ' -I' pathToReLeSrc 'include/rele/IRL' ...
    ' -I' pathToReLeSrc 'include/rele/solvers' ...
    ' ', pathToReLeLib, 'librele.a ' ...
    ' -largeArrayDims' ...
    ' -DARMA_BLAS_LONG' ...
    ' -lmwlapack -lmwblas -larmadillo' ...
    ' collectSamples.cpp CSDomainSettings.cpp'];
clc;
eval(mexCmd);


% %%
% clc
% mexCmd = ['mex -v -g GCC=''/usr/local/bin/gcc4.7'' CXX=''/usr/local/bin/g++4.7'' ' ...
%     ' -DARMA_DONT_USE_CXX11' ...
%     ' -largeArrayDims' ...
% ' -DARMA_BLAS_LONG' ...
% ' -lmwlapack -lmwblas -larmadillo' ...
% ' testCaseArma.cpp'];
% eval(mexCmd)

% testCaseArma