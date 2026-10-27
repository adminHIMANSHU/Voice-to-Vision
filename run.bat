@echo off
TITLE VoiceToVision 2.0 - Major Assistive AI Platform
COLOR 0B

echo =======================================================================
echo          VOICE TO VISION 2.0 - MAJOR ASSISTIVE AI PLATFORM
echo =======================================================================
echo.
echo Launching VoiceToVision with the active Python environment...
echo.

SET KMP_DUPLICATE_LIB_OK=TRUE

python app\main.py

echo.
echo =======================================================================
echo VoiceToVision session ended.
echo =======================================================================
pause
