How to install ReLe on windows using Visual Studio 2015
=======================================================
This guide is based on the [guide by keonkim](http://keon.io/mlpack-on-windows.html) that explains how to install [mlpack](http://mlpack.org/) on windows with Visual Studio 2015.

Prerequisites
-------------
The following tools are needed:

- CMake [Windows win64-x64 Installer](https://cmake.org/download/)
- Visual Studio 2015 (Visual Studio 2013 lacks some powerful C++11 features, that ReLe uses.)

Step 1: download ReLe
---------------------
First, create a project folder. For this example, I used `C:/projects/` folder and clone the current version of ReLe.
```
git clone https://github.com/AIRLab-POLIMI/ReLe.git
```
So the folder structure now becomes `C:/projects/ReLe/`.

Step 2: Installing Prerequisites
------------------------
### OpenBLAS
Open Visual Studio and click `File > New > Project from Existing Code`.

Select Visual C++ and select the file location (in this case, `C:/projects/ReLe/rele/`). Give any project name (can be anything, but I used rele for this example) and click Finish.

Next, it is time to download the dependencies using NuGet. NuGet is something like apt-get for debian. It makes things a lot easier and less time consuming.

Go to `Tools > NuGet Package Manager > Manage NuGet Packages for Solution` and click `Browse` tab.

I want to download OpenBLAS so search for it and click the project (rele) which you want to apply.

### Boost
Note that boost provides also prebuilt libraries [here](https://sourceforge.net/projects/boost/files/boost-binaries/).
Download the version named `msvc-14` that corresponds to Visual Studio 15 (e.g. [for boost 1.62 on win 64](https://sourceforge.net/projects/boost/files/boost-binaries/1.62.0/boost_1_62_0-msvc-14.0-64.exe/download)).
Execute the Boost installation script and replace the destination folder with `C:/projects/`.

### Armadillo
Download the latest stable version of [Armadillo](http://arma.sourceforge.net/download.html). As a courtesy. You extract it anywhere, but for this example I extracted it under `C:/projects/` folder. projects folder now contains `armadillo-7.500.0` and `ReLe`.

Under `armadillo-7.500.0`, create a new folder named `build`. And now go to command prompt by pressing windows button and typing `cmd`. Make sure you can use cmake in the command prompt. go to the `build` folder you just made, and copy and paste the following (all in one line).

```
cmake -G "Visual Studio 14 2015 Win64" -DBLAS_LIBRARY:FILEPATH="C:/projects/ReLe/rele/packages/OpenBLAS.0.2.14.1/lib/native/lib/x64/libopenblas.dll.a" -DLAPACK_LIBRARY:FILEPATH="C:/projects/ReLe/rele/packages/OpenBLAS.0.2.14.1/lib/native/lib/x64/libopenblas.dll.a" -DBUILD_SHARED_LIBS=OFF ..
```

If you used different directories other than ones used in this example, you must change the input accordingly.

Cmake then should create many files below build folder. click on `armadillo.sln` file to open with Visual Studio. Next, click `Build > Build Solution` to build armadillo. When you are done, close Visual Studio and go to the following step.

### NLopt
You can download the source code of nlopt from [here](http://ab-initio.mit.edu/wiki/index.php/NLopt#Download_and_installation).
To simplify installation, they provide a precompiled 32-bit and 64-bit Windows [DLLs](http://ab-initio.mit.edu/wiki/index.php/NLopt_on_Windows).

Extract the files into `C:/projects/`. You should have the directory `C:/projects/nlopt-2.4.2-dll64` (or 32 bit).

Move into the folder and run the following commands

```
call "%VS140COMNTOOLS%..\..\VC\vcvarsall.bat" x64
lib /def:libnlopt-0.def
```
*Attention:* replace x64 with x86 on Windows 32 bit with nlopt-2.4.2-dll32.

Step 3: Build ReLe
------------------

Now you can finally build ReLe!
The instructions are same as building armadillo, so it should be easier as you’ve already done it once.

Create a directory `build` in `C:/projects/ReLe/rele`. Move into the folder and run the following command
```
cmake -G "Visual Studio 14 2015 Win64" -DCMAKE_BUILD_TYPE=Debug -DBLAS_LIBRARY:FILEPATH="C:/projects/ReLe/rele/packages/OpenBLAS.0.2.14.1/lib/native/lib/x64/libopenblas.dll.a" -DLAPACK_LIBRARY:FILEPATH="C:/projects/ReLe/rele/packages/OpenBLAS.0.2.14.1/lib/native/lib/x64/libopenblas.dll.a" -DARMADILLO_INCLUDE_DIR="C:/projects/armadillo-7.500.0/include" -DARMADILLO_LIBRARY:FILEPATH="C:/projects/armadillo-7.500.0/build/Debug/armadillo.lib" -DBOOST_INCLUDEDIR:PATH="C:/projects/boost_1_62_0" -DBOOST_LIBRARYDIR:PATH="C:/projects/boost_1_62_0/lib64-msvc-14.0" -DNLOPT_INCLUDE_DIR:PATH="C:/projects/nlopt-2.4.2-dll64" -DNLOPT_LIBRARY:FILEPATH="C:/projects/nlopt-2.4.2-dll64/libnlopt-0.lib" ..
```

Cmake then should create many files below build folder. click on `rele.sln` file to open with Visual Studio. Next, click `Build > Build Solution` to build armadillo. This will compile the library and all the examples. You can easily build only the library by selecting `rele` in the solution explorer.
