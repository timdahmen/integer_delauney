@echo off
REM Build and install delauney (with the CUDA extension) in editable mode.
REM
REM cd to this script's own directory first: `pip install -e .` resolves "."
REM against the *shell's* working directory, so running this script from
REM elsewhere silently rebuilds whatever project the shell happens to be in.
cd /d "%~dp0"

REM Resolve the newest Visual Studio carrying the C++ toolset; this project is
REM built on machines with different Visual Studio versions installed.
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
set "VSPATH="
for /f "usebackq delims=" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSPATH=%%i"
if not defined VSPATH (
    echo ERROR: no Visual Studio with the C++ toolset found.
    exit /b 1
)

REM nvcc rejects MSVC toolsets newer than the CUDA release knows about, so pin
REM 14.44 where it is installed and take the default otherwise.
set "VCVARS_ARGS="
for /d %%d in ("%VSPATH%\VC\Tools\MSVC\14.44*") do set "VCVARS_ARGS=-vcvars_ver=14.44"

call "%VSPATH%\VC\Auxiliary\Build\vcvars64.bat" %VCVARS_ARGS%
echo === building delauney in %CD% ===
pip install -e . --no-cache-dir
