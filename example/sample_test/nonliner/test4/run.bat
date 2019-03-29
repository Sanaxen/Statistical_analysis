set PYPATH=%USERPROFILE%\Anaconda3


set LDM=..\..\..\all_build\x64\Release
copy ..\..\..\..\third_party\bin\*.dll

set progress=1
if "%1"=="0" set progress=0

del /Q images\*.png

%LDM%\TimeSeriesRegression.exe  --x 0 --y 1 --y_var 1 --t_var 0 --csv sample.csv --learning_rate 1 --opt_type adam ^
--test 0.1 --progress %progress% --plot 5 --tol 0.03  --header 1 ^
--epochs 26000 --n_layers 6  --n_rnn_layers 1  --normal minmax  --rnn_type lstm ^
--seq_len 14 --minibatch_size 14 --hidden_size 64 --test_mode 0 --early_stopping 1 --test_mode 0 --support 0


del *.dll

