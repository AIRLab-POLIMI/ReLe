### How to install ReLe under Windows ###

This guide is out-of-date, please refert to [this](https://github.com/AIRLab-POLIMI/ReLe/README_WINVS.md).

#Prerequisites
- Mingw 32 bits or 64 bits
MinGW 32 bits
http://www.mingw.org/
MinGW 64 bits
http://sourceforge.net/projects/mingwbuilds/files/host-windows/releases/4.8.1/64-bit/threads-posix/seh/x64-4.8.1-release-posix-seh-rev5.7z/download
- CMake
http://www.cmake.org/
Note: during the install phase select the option "Add to path for all the users"

#MinGW 64 bits installation
Download the compressed archive and extract the folder in the root ("C:\")
Add the folder ";C:\mingw64\bin" to the system path (note that the ; is the path separator)

# Compile blas and atlas libraries
http://icl.cs.utk.edu/lapack-for-windows/lapack/#build
1. Extract the archive
2. Create a "build" folder into lapack folder
3. Open cmake
	3.1 Point to your lapack folder as the source code folder
	3.2 Use the "build" folder as build folder
	3.3 Click configure, check the install path if you want to have the libraries and includes in a particular location.
		We suggest to use "C:/usr" as installation path
	3.4 Choose MinGW Makefiles.
	3.5 Set the 'BUILD_SHARED_LIBS' option to ON.
	3.6 Set the 'CMAKE_GNUtoMS' option to ON.
	3.7 Click again configure until everything becomes white
	3.8 Click generate, that will create the mingw build.
	3.9 Close CMake
4. Open a cmd prompt (Click Run.. then enter cmd)
5. Go to your build folder using the cd command
6. type mingw32-make
7. type mingw32-make install
8. You must have the libraries installed in C:/usr
9. Add to the system path the folder ";C:\usr\bin"

# Compile armadillo
http://arma.sourceforge.net/download.html
1. Extract the archive
2. Open cmake
	2.1 Select armadillo folder as source code folder
	2.2 Select the same folder as build folder
	2.3 Click configure
	2.4 Choose MinGW Makefiles.
	2.5 Click again configure until everything becomes white
	2.6 Click generate, that will create the mingw build.
	2.7 Close CMake
3. Open a cmd prompt (Click Run.. then enter cmd)
4. Go to your armadillo folder using the cd command
5. Type mingw32-make
7. Type mingw32-make install
8. You must have the libraries installed in C:/usr

# Compile ReLe
1. Extract the archive
2. Open CMake
	2.1 Select ReLe-master/rele folder as source code folder
	2.2 Create a folder ReLe-master/build as build folder
	2.3 Click configure
	2.4 Choose MinGW Makefiles.
	2.5 Click again configure until everything becomes white
	2.6 Click generate, that will create the mingw build.
	2.7 Close CMake
3. Open a cmd prompt (Click Run.. then enter cmd)
4. Go to your ReLe-master/build folder using the cd command
5. Type mingw32-make
8. You must have a library named "librele.a" and a file "lqr_pgpe.exe"
9. Try to execute lqr_pgpe.exe
