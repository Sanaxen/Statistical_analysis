set PYPATH=%USERPROFILE%\Anaconda3

set LDM=..\..\..\all_build\x64\Release
copy ..\..\..\..\third_party\bin\*.dll

: create x,y,z,w csv files
%PYPATH%\python.exe Untitled.py

%LDM%\LiNGAM.exe --csv sample.csv --header 0
call gr.bat
copy Digraph.png Digraph1.png /v /y

%LDM%\LiNGAM.exe --csv sample2.csv --header 1
call gr.bat
copy Digraph.png Digraph2.png /v /y

#call gr.bat
:pause
del *.dll