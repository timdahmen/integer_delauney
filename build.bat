@echo off
REM Build and install delauney (with the CUDA extension) in editable mode.
REM
REM cd to this script's own directory first: `pip install -e .` resolves "."
REM against the *shell's* working directory, so running this script from
REM elsewhere silently rebuilds whatever project the shell happens to be in.
cd /d "%~dp0"

call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
echo === building delauney in %CD% ===
pip install -e . --no-cache-dir
