set PYPATH=%USERPROFILE%\Anaconda3

set LDM=..\..\..\all_build\x64\Release
copy ..\..\..\..\third_party\bin\*.dll

%LDM%\multi_scatter.exe --csv test.csv

%LDM%\multi_scatter.exe --csv test.csv --col1 0 --col2 1 --palette "rgbformulae 21, 22, 23" --linear_regression 1 --ellipse 1 --grid 10

del *.dll