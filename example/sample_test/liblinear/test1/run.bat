set PYPATH=%USERPROFILE%\Anaconda3

set LDM=..\..\..\all_build\x64\Release
copy ..\..\..\..\third_party\bin\*.dll


%LDM%\liblinear.exe --csv sample-data.csv --header 1  --x_var "年齢" --x_var "血圧"  --x_var "体重" --y_var "病気"

train -s 0  -B 5 sample-data.csv_libsvm.format


predict -b 1 sample-data.csv_libsvm.format sample-data.csv_libsvm.format.model output.txt

:pause
del *.dll