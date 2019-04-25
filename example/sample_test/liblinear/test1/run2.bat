set PYPATH=%USERPROFILE%\Anaconda3

set LDM=..\..\..\all_build\x64\Release
copy ..\..\..\..\third_party\bin\*.dll


%LDM%\logistic_regression_train.exe --L2 0.000001 --file sample-data.csv_libsvm.format --model model
%LDM%\logistic_regression_predict.exe --file sample-data.csv_libsvm.format --model model --output out.txt


:pause
del *.dll