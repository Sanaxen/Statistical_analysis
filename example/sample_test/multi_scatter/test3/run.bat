set PYPATH=%USERPROFILE%\Anaconda3

set LDM=..\..\..\all_build\x64\Release
copy ..\..\..\..\third_party\bin\*.dll

%LDM%\multi_scatter.exe --header 1 --col 1 --csv Boston.csv --x_var "C" --x_var "E" --y_var "B" --y_var "A"  --palette "rgbformulae 21, 22, 23" --linear_regression 1 --ellipse 1 --grid 10

del *.dll