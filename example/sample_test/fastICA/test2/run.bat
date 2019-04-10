set PYPATH=%USERPROFILE%\Anaconda3


set LDM=..\..\..\all_build\x64\Release
copy ..\..\..\..\third_party\bin\*.dll

:wav -> csv
%PYPATH%\python.exe Untitled.py

:ICA
%LDM%\fastICA_test2.exe --csv sample.csv --x_var 0 --x_var 1 --x_var 2

:csv -> wav
%PYPATH%\python.exe Untitled1.py


del *.dll