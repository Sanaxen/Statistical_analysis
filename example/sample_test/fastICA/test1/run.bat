set PYPATH=%USERPROFILE%\Anaconda3

set LDM=..\..\..\all_build\x64\Release

copy ..\..\..\..\third_party\bin\*.dll

: create mix signals
%PYPATH%\python.exe Untitled.py

%LDM%\fastICA_test1.exe

del *.dll
: > ica.dat
