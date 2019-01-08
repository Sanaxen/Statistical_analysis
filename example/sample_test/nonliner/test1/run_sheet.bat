set PYPATH=%USERPROFILE%\Anaconda3

set LDM=..\..\..\all_build\x64\Release

copy ..\..\..\..\third_party\bin\*.dll

set progress=1
if "%1"=="0" set progress=0

del /Q images\*.png

%LDM%\NonLinearRegression.exe  --header 1 --x 1 --y 1 --x_var x --y_var y --csv sheet11.csv ^
--test 0 --tol 0.005 --progress %progress%  --plot 5 --epochs 800  --minibatch_size 64 ^
--learning_rate 1 --n_layers 10 --normal minmax --dec_random 0.0 --fluctuation 0.1

goto end

%LDM%\NonLinearRegression.exe  --header 1 --x 1 --y 1 --x_var x --y_var y --csv sheet10.csv ^
--test 0 --tol 0.005 --progress %progress%  --plot 5 --epochs 800  --minibatch_size 64 ^
--learning_rate 1 --n_layers 10 --normal minmax

goto end

%LDM%\NonLinearRegression.exe  --header 1 --x 1 --y 1 --x_var x --y_var y --csv sheet9.csv ^
--test 0 --tol 0.005 --progress %progress%  --plot 5 --epochs 800  --minibatch_size 64 ^
--learning_rate 1 --n_layers 10 --normal minmax

goto end

%LDM%\NonLinearRegression.exe  --header 1 --x 1 --y 1 --x_var x --y_var y --csv sheet8.csv ^
--test 0 --tol 0.005 --progress %progress%  --plot 5 --epochs 800  --minibatch_size 64 ^
--learning_rate 1 --n_layers 10 --normal minmax

goto end

:end

del *.dll

