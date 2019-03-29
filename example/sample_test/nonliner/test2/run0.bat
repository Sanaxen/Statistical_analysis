set PYPATH=%USERPROFILE%\Anaconda3


set LDM=..\..\..\all_build\x64\Release
copy ..\..\..\..\third_party\bin\*.dll

set progress=1
if "%1"=="0" set progress=0

del /Q images\*.png

%LDM%\TimeSeriesRegression.exe  --x 0 --y 1 --y_var 2 --t_var "0" --csv sample.csv --learning_rate 1 --opt_type adam ^
--test 0.3 --progress %progress% --plot 5 --tol 0.0005 ^
--epochs 26000 --n_layers 5  --n_rnn_layers 1 --rnn_type lstm ^
--seq_len 100 --minibatch_size 100 --hidden_size 6 --test_mode 0 --normal zscore ^
--early_stopping 0 --prophecy 2000 --test_mode 0
: > log1.txt
goto end

%LDM%\TimeSeriesRegression.exe  --x 0 --y 1 --y_var 2 --t_var "0" --csv sample.csv --learning_rate 1 --opt_type adam ^
--test 0.3 --progress %progress% --plot 5 --tol 0.0005 ^
--epochs 26000 --n_layers 5  --n_rnn_layers 1 --rnn_type lstm ^
--seq_len 100 --minibatch_size 100 --hidden_size 20 --test_mode 0 --normal zscore ^
--early_stopping 0 --prophecy 2000 --test_mode 0
: > log1.txt
goto end


ffmpeg -y -i "images/test_%%04d.png" -r 2 -crf 30 bbb.mp4
ffmpeg -y -i "images/test_%%04d.png" -r 2 -crf 30 bbb.flv
ffmpeg -y -i "images/test_%%04d.png" -an -r 10  -pix_fmt rgb24 -vf "setpts=2.0*PTS"  -vf "scale=500:-1" -f gif bbb.gif


:end

del *.dll

