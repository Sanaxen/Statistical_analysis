set PYPATH=%USERPROFILE%\Anaconda3

set LDM=..\..\..\all_build\x64\Release
copy ..\..\..\..\third_party\bin\*.dll

%LDM%\multi_scatter.exe --csv Boston.csv  --header 1 --col 1

:pause

%LDM%\multi_scatter.exe --csv Boston.csv  --header 1 --col 1 --col1 7 --col2 5 --palette "rgbformulae 4, 4, 4"
%LDM%\multi_scatter.exe --csv Boston.csv  --header 1 --col 1 --col1 7 --col2 5 --palette "rgbformulae 30, 31, 32"


%LDM%\multi_scatter.exe --csv Boston.csv  --header 1 --col 1 --col1 7 --col2 5 --palette "rgbformulae 4, 4, 4" --linear_regression 1
del *.dll