set PYPATH=%USERPROFILE%\Anaconda3

set LDM=..\..\..\all_build\x64\Release
copy ..\..\..\..\third_party\bin\*.dll

%LDM%\multi_scatter.exe --csv xydata.csv

%LDM%\multi_scatter.exe --csv xydata.csv --col1 0 --col2 1 --palette "rgbformulae 22, 13, -31" --linear_regression 1 --ellipse 1 --header 1 --col 1 --grid 30
:%LDM%\multi_scatter.exe --csv xydata.csv --col1 0 --col2 1  --linear_regression 1 --ellipse 1 --header 1 --col 1

del *.dll