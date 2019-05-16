:
set LDM=..\..\..\all_build\x64\Release
copy ..\..\..\..\third_party\bin\*.dll

%LDM%\TimeSeriesRegression.exe --@ comman_args.txt

:end

del *.dll