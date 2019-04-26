set PYPATH=%USERPROFILE%\Anaconda3

set LDM=..\..\..\all_build\x64\Release
copy ..\..\..\..\third_party\bin\*.dll

:%LDM%\liblinear.exe --csv sample-data.csv --header 1  --x_var "�N��" --x_var "�x����" --x_var "����"  --x_var "�̏d" --x_var "����" --y_var "�a�C"
%LDM%\liblinear.exe --csv sample-data.csv --header 1 --col 0 --y_var 5 --x_var 1 --x_var 2 --x_var 3 --x_var 4 --x_var 6 --fold_cv 0

%LDM%\logistic_regression.exe --train 1 --L2 1.0 --file sample-data.csv_libsvm.train --model model --bias 0
%LDM%\logistic_regression.exe --predict 1 --file sample-data.csv_libsvm.train --model model --output out.txt


:pause
del *.dll