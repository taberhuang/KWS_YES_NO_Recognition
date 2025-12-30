@echo off
REM Build script for real-time keyword spotting with microphone input
REM Requires: MinGW64, PortAudio

echo === Building Real-time Keyword Spotting ===

REM Set MinGW64 path (adjust if needed)
set MINGW_PATH=C:\msys64\mingw64
set PATH=%MINGW_PATH%\bin;%PATH%

REM Set PortAudio path (adjust to your installation)
REM Option 1: If you installed via MSYS2: pacman -S mingw-w64-x86_64-portaudio
set PORTAUDIO_DIR=%MINGW_PATH%

echo Using MinGW from: %MINGW_PATH%
echo Using PortAudio from: %PORTAUDIO_DIR%

REM Create build directory
if not exist build_realtime mkdir build_realtime

REM Backup original CMakeLists.txt if exists
if exist CMakeLists.txt.bak del CMakeLists.txt.bak
rename CMakeLists.txt CMakeLists.txt.bak

REM Use realtime CMakeLists
copy /Y CMakeLists_realtime.txt CMakeLists.txt

cd build_realtime

REM Run CMake
cmake -G "MinGW Makefiles" ^
    -DCMAKE_C_COMPILER=%MINGW_PATH%\bin\gcc.exe ^
    -DCMAKE_CXX_COMPILER=%MINGW_PATH%\bin\g++.exe ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DPORTAUDIO_DIR=%PORTAUDIO_DIR% ^
    ..

if errorlevel 1 (
    echo CMake configuration failed!
    cd ..
    REM Restore original CMakeLists.txt
    del CMakeLists.txt
    rename CMakeLists.txt.bak CMakeLists.txt
    exit /b 1
)

REM Build
cmake --build . -j%NUMBER_OF_PROCESSORS%

cd ..

REM Restore original CMakeLists.txt
del CMakeLists.txt
rename CMakeLists.txt.bak CMakeLists.txt

if errorlevel 1 (
    echo Build failed!
    exit /b 1
)

echo.
echo === Build successful! ===
echo Executable: build_realtime\kws_realtime.exe
echo.
echo Usage: kws_realtime.exe [device_number]
echo   Run without arguments to use default microphone
echo   Run with device number to select specific input device
echo.
