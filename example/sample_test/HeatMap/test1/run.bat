set PYPATH=%USERPROFILE%\Anaconda3

set LDM=..\..\..\all_build\x64\Release
copy ..\..\..\..\third_party\bin\*.dll


:%LDM%\heatmap.exe --csv Boston.csv  --header 1 --col_index 0
:%LDM%\heatmap.exe --csv test.csv  --header 1
:%LDM%\heatmap.exe --csv test2.csv

%LDM%\heatmap.exe --csv 2-3b.csv --header 1 --col_index 1 --col 2

del *.dll