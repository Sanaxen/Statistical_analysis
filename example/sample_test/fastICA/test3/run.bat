set PYPATH=%USERPROFILE%\Anaconda3


set LDM=..\..\..\all_build\x64\Release
copy ..\..\..\..\third_party\bin\*.dll

:wav -> csv
%PYPATH%\python.exe Untitled.py

:ICA
%LDM%\fastICA_test2.exe

:csv -> wav
%PYPATH%\python.exe Untitled1.py


del *.dll