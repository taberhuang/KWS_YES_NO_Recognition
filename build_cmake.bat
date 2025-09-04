@echo off
setlocal enabledelayedexpansion

REM Set MinGW64 paths first
set "MINGW64_PATH=C:\msys64\mingw64"
set "PATH=%MINGW64_PATH%\bin;%SystemRoot%\system32;%SystemRoot%"

REM Default parameter handling
if "%1"=="" (
    set COMMAND=build_run
) else (
    set COMMAND=%1
)

echo KWS Yes/No Recognition Build Script
echo ===================================
echo Command: %COMMAND%
echo MinGW64 Path: %MINGW64_PATH%

REM Check required tools
call :check_tools
if %errorlevel% neq 0 goto :end

REM Execute based on command
if "%COMMAND%"=="build" goto :do_build
if "%COMMAND%"=="rebuild" goto :do_rebuild
if "%COMMAND%"=="build_run" goto :do_build_run
if "%COMMAND%"=="rebuild_run" goto :do_rebuild_run
if "%COMMAND%"=="run" goto :do_run
if "%COMMAND%"=="clean" goto :do_clean

REM Default case: show help
echo Usage: %0 [command]
echo Commands:
echo   (empty)    - build and run tests (default)
echo   build      - build only
echo   rebuild    - clean and rebuild
echo   build_run  - build and run tests
echo   rebuild_run- clean, rebuild and run tests
echo   run        - run tests only
echo   clean      - clean build directory
goto :end

:do_build
call :build
goto :end

:do_rebuild
call :rebuild
goto :end

:do_build_run
call :build
if %errorlevel% equ 0 (
    call :run_tests
)
goto :end

:do_rebuild_run
call :rebuild
if %errorlevel% equ 0 (
    call :run_tests
)
goto :end

:do_run
call :run_tests
goto :end

:do_clean
call :clean
goto :end

:check_tools
echo Checking required tools...

REM Check Ninja
where ninja >nul 2>nul
if %errorlevel% neq 0 (
    echo ERROR: Ninja not found. Please install Ninja build system.
    echo Download from: https://github.com/ninja-build/ninja/releases
    exit /b 1
)

REM Check CMake
where cmake >nul 2>nul
if %errorlevel% neq 0 (
    echo ERROR: CMake not found. Please install CMake.
    echo Download from: https://cmake.org/download/
    exit /b 1
)

REM Check GCC compiler
where gcc >nul 2>nul
if %errorlevel% neq 0 (
    echo ERROR: GCC compiler not found in PATH.
    echo Please ensure MinGW64 is installed and in PATH.
    echo Current PATH includes: %PATH%
    exit /b 1
)

REM Check G++ compiler
where g++ >nul 2>nul
if %errorlevel% neq 0 (
    echo ERROR: G++ compiler not found in PATH.
    echo Please ensure MinGW64 is installed and in PATH.
    exit /b 1
)

echo All tools found.
echo GCC: & gcc --version | findstr "gcc"
goto :eof

:build
echo Building KWS Yes/No Recognition...

REM Create build directory
if not exist "build_cmake" mkdir build_cmake

REM Enter build directory
cd build_cmake

REM Ensure MinGW path is set for CMake subprocess
set "PATH=%MINGW64_PATH%\bin;%PATH%"

echo Configuring with CMake...
echo Using compilers from: %MINGW64_PATH%\bin
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release ^
    -DCMAKE_C_COMPILER="%MINGW64_PATH%\bin\gcc.exe" ^
    -DCMAKE_CXX_COMPILER="%MINGW64_PATH%\bin\g++.exe"
if %errorlevel% neq 0 (
    echo ERROR: CMake configuration failed.
    cd ..
    exit /b 1
)

echo Building with Ninja...
ninja
if %errorlevel% neq 0 (
    echo ERROR: Build failed.
    cd ..
    exit /b 1
)

echo Build completed successfully!
echo Executable: build_cmake\kws_yesno.exe

cd ..
goto :eof

:rebuild
echo Rebuilding KWS Yes/No Recognition...

REM Clean build directory
if exist "build_cmake" rmdir /s /q build_cmake

REM Rebuild
call :build
goto :eof

:run_tests
echo Running tests...

REM Check if executable exists
if not exist "build_cmake\kws_yesno.exe" (
    echo ERROR: Executable not found. Please build first.
    exit /b 1
)

REM Check if test files exist
if not exist "yes_1000ms.wav" (
    echo ERROR: Test file yes_1000ms.wav not found.
    exit /b 1
)

if not exist "no_1000ms.wav" (
    echo ERROR: Test file no_1000ms.wav not found.
    exit /b 1
)

echo ====================================
echo Testing YES recognition:
echo ====================================
build_cmake\kws_yesno.exe yes_1000ms.wav

echo ====================================
echo Testing NO recognition:
echo ====================================
build_cmake\kws_yesno.exe no_1000ms.wav

echo Tests completed!
goto :eof

:clean
echo Cleaning build directory...
if exist "build_cmake" (
    rmdir /s /q build_cmake
    echo Build directory cleaned.
) else (
    echo Build directory not found.
)
goto :eof

:end
