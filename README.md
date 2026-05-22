# integer_delauney
A GPU Implementation of a Delauney Triangulation for Pixel Coordinates

# Fixing PATH issue with VS2019 Professional
VS 2019 Professional is at C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional. Run this in your PowerShell terminal to load the MSVC environment and then install:
`cmd /c '"C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\VC\Auxiliary\Build\vcvarsall.bat" x64 && set' | ForEach-Object { if ($_ -match "^([^=]+)=(.*)$") { [System.Environment]::SetEnvironmentVariable($Matches[1], $Matches[2]) } }`

Then:
`D:\<yourpath>\.venv\Scripts\pip install -e .`