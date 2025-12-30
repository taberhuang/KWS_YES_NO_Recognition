@echo off
echo === Testing saved WAV files ===
echo.

for %%i in (build_realtime\test_*.wav) do (
    for /f "tokens=*" %%r in ('kws_0_9.exe %%i ^| findstr "FINAL RESULT"') do (
        echo %%~nxi: %%r
    )
)

echo.
echo === Done ===
pause
