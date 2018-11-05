cd script

:set LDM=..\..\..\all_build\x64\Release
:copy ..\..\..\..\third_party\bin\*.dll

set LDM=..
%LDM%\Heatmap.exe --csv ..\sample\2-3b.csv  --header 1 --col 2 --col_index 1


cd ..
:del *.dll