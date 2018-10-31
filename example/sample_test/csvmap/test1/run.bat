set PYPATH=%USERPROFILE%\Anaconda3

set LDM=..\..\..\all_build\x64\Release
copy ..\..\..\..\third_party\bin\*.dll


%LDM%\csvmap.exe --csv Boston.csv  --header 0 --col_index 0
: --palette "rgbformulae 22, 13, -31"

del *.dll