@echo off
echo Setting up MSVC environment for CUDA development...

:: Detect latest Visual Studio installation (modify paths as needed)
if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" (
    call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
) else if exist "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvarsall.bat" (
    call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
) else (
    echo ERROR: Could not find Visual Studio installation.
    echo Please install Visual Studio with C++ development tools.
    exit /b 1
)

echo MSVC environment is now ready.
echo You can now compile your CUDA code.