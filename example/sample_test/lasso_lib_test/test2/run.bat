set PYPATH=%USERPROFILE%\Anaconda3

set LDM=..\..\..\all_build\x64\Release
copy ..\..\..\..\third_party\bin\*.dll

del sample.csv
%LDM%\lasso_test.exe
%LDM%\lasso_test.exe --csv sample.csv --L1 1.0 --header 1

:pause

del *.dll