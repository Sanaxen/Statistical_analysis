set PYPATH=%USERPROFILE%\Anaconda3


set LDM=..\..\..\all_build\x64\Release
copy ..\..\..\..\third_party\bin\*.dll

set progress=1
if "%1"=="0" set progress=0

del /Q images\*.png
:%LDM%\TimeSeriesRegression.exe --header 1  --x 2 --y 3 --x_var "t1" --x_var "t2" --y_var "y1" --y_var "y2" --y_var "y3" --csv sample2.csv --test 0.5 --progress %progress% --plot 50 --tol 0.014


%LDM%\TimeSeriesRegression.exe  --x 0 --y 3 --y_var 2 --y_var 3 --y_var 4 --t_var "0" --csv sample.csv --learning_rate 1 --opt_type adam ^
--test 0.5 --progress %progress% --plot 5 --tol 0.01 ^
--epochs 26000 --n_layers 6  --n_rnn_layers 1 --rnn_type lstm ^
--seq_len 24 --minibatch_size 48 --hidden_size 64 --test_mode 0 --normal zscore --early_stopping 0 > log1.txt
goto end


%LDM%\TimeSeriesRegression.exe  --x 0 --y 3 --y_var 2 --y_var 3 --y_var 4 --t_var "0" --csv sample.csv --learning_rate 1 --opt_type adam ^
--test 0.5 --progress %progress% --plot 5 --tol 0.01 ^
--epochs 26000 --n_layers 1  --n_rnn_layers 3 --rnn_type lstm ^
--seq_len 24 --minibatch_size 48 --hidden_size 64 --test_mode 0 --normal zscore --early_stopping 0: > log2.txt
:goto end

%LDM%\TimeSeriesRegression.exe  --x 0 --y 3 --y_var 2 --y_var 3 --y_var 4 --t_var "0" --csv sample.csv --learning_rate 1 --opt_type adam ^
--test 0.5 --progress %progress% --plot 5 --tol 0.01 ^
--epochs 26000 --n_layers 2  --n_rnn_layers 4 --rnn_type lstm ^
--seq_len 24 --minibatch_size 24 --hidden_size 64 --test_mode 0 --normal zscore > log3.txt
:goto end



%LDM%\TimeSeriesRegression.exe  --x 0 --y 3 --y_var 2 --y_var 3 --y_var 4 --t_var "0" --csv sample.csv --learning_rate 0.1 --opt_type adam ^
--test 0.5 --progress %progress% --plot 5 --tol 0.01 ^
--epochs 26000 --n_layers 6  --n_rnn_layers 3 ^
--seq_len 24 --minibatch_size 32 --hidden_size 64 --test_mode 0  --normal zscore > log4.txt

%LDM%\TimeSeriesRegression.exe  --x 0 --y 3 --y_var 2 --y_var 3 --y_var 4 --t_var "0" --csv sample.csv --learning_rate 0.1 --opt_type adam ^
--test 0.5 --progress %progress% --plot 5 --tol 0.01 ^
--epochs 26000 --n_layers 8  --n_rnn_layers 3 ^
--seq_len 24 --minibatch_size 32 --hidden_size 64 --test_mode 0  --normal zscore > log5.txt

:goto end
: NG!!
:%LDM%\TimeSeriesRegression.exe  --x 0 --y 3 --y_var 2 --y_var 3 --y_var 4 --t_var "0" --csv sample.csv --learning_rate 0.1 --opt_type adam ^
:--test 0.5 --progress %progress% --plot 5 --tol 0.03 ^
:--epochs 26000 --n_layers 24  --n_rnn_layers 4 ^
:--seq_len 24 --minibatch_size 32 --hidden_size 64 --test_mode 0  --normal zscore  > log6.txt

%LDM%\TimeSeriesRegression.exe  --x 0 --y 3 --y_var 2 --y_var 3 --y_var 4 --t_var "0" --csv sample.csv --learning_rate 0.1 --opt_type adam ^
--test 0.5 --progress %progress% --plot 5 --tol 0.03 ^
--epochs 26000 --n_layers 6  --n_rnn_layers 1 ^
--seq_len 24 --minibatch_size 32 --hidden_size 64 --test_mode 0  --normal zscore  > log7.txt

%LDM%\TimeSeriesRegression.exe  --x 0 --y 3 --y_var 2 --y_var 3 --y_var 4 --t_var "0" --csv sample.csv --learning_rate 0.1 --opt_type adam ^
--test 0.5 --progress %progress% --plot 5 --tol 0.01 ^
--epochs 26000 --n_layers 8  --n_rnn_layers 2 ^
--seq_len 24 --minibatch_size 32 --hidden_size 64 --test_mode 0  --normal zscore  > log8.txt

ffmpeg -y -i "images/test_%%04d.png" -r 2 -crf 30 bbb.mp4
ffmpeg -y -i "images/test_%%04d.png" -r 2 -crf 30 bbb.flv
ffmpeg -y -i "images/test_%%04d.png" -an -r 10  -pix_fmt rgb24 -vf "setpts=2.0*PTS"  -vf "scale=500:-1" -f gif bbb.gif


:end

del *.dll

