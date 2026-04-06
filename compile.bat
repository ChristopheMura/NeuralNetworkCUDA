@echo off
echo ========================================
echo     Build CUDA Project - Release
echo ========================================

:: Se placer a la racine du projet (dossier parent de build/)
cd /d "%~dp0"

:: Creer le dossier build s'il n'existe pas
if not exist "build" (
    echo Creation du dossier build...
    mkdir build
)

cd build

:: Generation des fichiers CMake
echo.
echo [1/2] Generation CMake - Visual Studio 17 2022 x64...
cmake -G "Visual Studio 17 2022" -A x64 ..

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERREUR] La generation CMake a echoue.
    pause
    exit /b 1
)

:: Compilation
echo.
echo [2/2] Compilation en mode Release...
cmake --build . --config Release

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERREUR] La compilation a echoue.
    pause
    exit /b 1
)

echo.
echo ========================================
echo     Build termine avec succes !
echo     Executable : build\Release\run.exe
echo ========================================

pause