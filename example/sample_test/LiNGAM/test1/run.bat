set PYPATH=%USERPROFILE%\Anaconda3

set LDM=..\..\..\all_build\x64\Release
copy ..\..\..\..\third_party\bin\*.dll

: create x,y,z,w csv files
%PYPATH%\python.exe Untitled.py

%LDM%\LiNGAM.exe

call gr.bat
:pause
del *.dll