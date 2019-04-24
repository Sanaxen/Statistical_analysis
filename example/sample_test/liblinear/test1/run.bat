set PYPATH=%USERPROFILE%\Anaconda3

set LDM=..\..\..\all_build\x64\Release
copy ..\..\..\..\third_party\bin\*.dll


%LDM%\liblinear.exe --csv sample-data.csv --header 1  --x_var "�N��" --x_var "����"  --x_var "�̏d" --y_var "�a�C"

train -s 0  -B 5 sample-data.csv_libsvm.format


predict -b 1 sample-data.csv_libsvm.format sample-data.csv_libsvm.format.model output.txt

:pause
del *.dll