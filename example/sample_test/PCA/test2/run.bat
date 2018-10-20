set PYPATH=%USERPROFILE%\Anaconda3

set LDM=..\..\..\all_build\x64\Release
copy ..\..\..\..\third_party\bin\*.dll

%LDM%\PCA_test2.exe --csv 2-3c.csv --header 1 --col 1

:pause
del *.dll