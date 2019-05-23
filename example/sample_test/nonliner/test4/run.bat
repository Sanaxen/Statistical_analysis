set PYPATH=%USERPROFILE%\Anaconda3


set LDM=..\..\..\all_build\x64\Release
copy ..\..\..\..\third_party\bin\*.dll

set progress=1
if "%1"=="0" set progress=0

del /Q images\*.png
:%LDM%\TimeSeriesRegression.exe --header 1  --x 2 --y 3 --x_var "t1" --x_var "t2" --y_var "y1" --y_var "y2" --y_var "y3" --csv sample2.csv --test 0.5 --progress %progress% --plot 50 --tol 0.014

%LDM%\TimeSeriesRegression.exe  --x 0 --y 1 --y_var 1 --t_var 0 --csv sample.csv --learning_rate 0.1 --opt_type adam ^
--test 0.05 --progress %progress% --plot 5 --tol 0.03  --header 1 ^
--epochs 26000 --n_layers 16  --n_rnn_layers 2  --normal minmax  --rnn_type lstm --use_cnn 2 ^
--seq_len 34 --minibatch_size 34 --hidden_size 200 --test_mode 0 --early_stopping 1 --test_mode 0


del *.dll

