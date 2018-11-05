cd script

:set LDM=..\..\..\all_build\x64\Release
:copy ..\..\..\..\third_party\bin\*.dll

set LDM=..
%LDM%\multi_scatter.exe --csv ..\sample\Boston.csv  --header 1 --col 1

:pause

%LDM%\multi_scatter.exe --csv ..\sample\Boston.csv  --header 1 --col 1 --col1 7 --col2 5 --palette "rgbformulae 4, 4, 4"

cd ..
:del *.dll