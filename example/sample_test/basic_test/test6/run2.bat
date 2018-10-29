set PYPATH=%USERPROFILE%\Anaconda3


set LDM=..\..\..\all_build\x64\Release
copy ..\..\..\..\third_party\bin\*.dll

:%LDM%\test6.exe --header 1  --csv winequality-red.csv > log.txt

:%LDM%\test6.exe --header 1  --csv winequality-red.csv --x_var density --y_var alcohol > log.txt
:type log.txt

%LDM%\test6.exe --header 1  --csv winequality-red.csv  --y_var "quality" --normalize 1 > log.txt
type log.txt

:pause


del *.dll