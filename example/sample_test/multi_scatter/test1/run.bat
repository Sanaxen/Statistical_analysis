set PYPATH=%USERPROFILE%\Anaconda3

set LDM=..\..\..\all_build\x64\Release
copy ..\..\..\..\third_party\bin\*.dll

%LDM%\multi_scatter.exe --csv Boston.csv  --header 1 --col 1

:pause

%LDM%\multi_scatter.exe --csv Boston.csv  --header 1 --col 1 --col1 2 --col2 1

del *.dll