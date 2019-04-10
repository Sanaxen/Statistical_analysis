set PYPATH=%USERPROFILE%\Anaconda3


set LDM=..\..\..\all_build\x64\Release
copy ..\..\..\..\third_party\bin\*.dll

%LDM%\lineplot.exe --header 1 --csv sample.csv  --y_var y1 --y_var y2 --y_var y3 --max_lines 20
:%LDM%\lineplot.exe --header 1 --csv sample.csv --x_var X --y_var y1 --y_var y2 --y_var y3 --max_lines 20



del *.dll