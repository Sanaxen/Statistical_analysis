set PYPATH=%USERPROFILE%\Anaconda3


set LDM=..\..\..\all_build\x64\Release
copy ..\..\..\..\third_party\bin\*.dll

set progress=1
if "%1"=="0" set progress=0

del /Q images\*.png

%LDM%\TimeSeriesRegression.exe  --x 0 --y 1 --y_var D1 --t_var 0 --csv qtdbsel102_train.csv --learning_rate 0.5 --opt_type adam ^
--test 0.2 --progress %progress% --plot 5 --tol 0.001  --header 1 ^
--epochs 26000 --n_layers 10  --n_rnn_layers 1  --normal zscore  --rnn_type lstm --use_cnn 0 ^
--seq_len 30 --minibatch_size 30 --hidden_size 20 --early_stopping 1 --test_mode 0 ^
--out_seq_len 3

goto end

:end
del *.dll

