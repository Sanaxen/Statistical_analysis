set PYPATH=%USERPROFILE%\Anaconda3


set LDM=..\..\..\all_build\x64\Release
copy ..\..\..\..\third_party\bin\*.dll

set progress=1
if "%1"=="0" set progress=0

del /Q images\*.png


%LDM%\TimeSeriesRegression.exe  --x 0 --y 3 --y_var 2 --y_var 3 --y_var 4 --t_var "0" ^
--csv sample.csv --learning_rate 0.1 --opt_type adam ^
--test 0.2 --progress 0 --plot 5 --tol 0.001 ^
--epochs 26000 --n_layers 1  --n_rnn_layers 3 --rnn_type lstm ^
--prophecy 2000 ^
--out_seq_len 10 --use_cnn 1 ^
--seq_len 24 --minibatch_size 48 --hidden_size 128 --test_mode 0 --normal zscore --early_stopping 0
goto end

ffmpeg -y -i "images/test_%%04d.png" -r 2 -crf 30 bbb.mp4
ffmpeg -y -i "images/test_%%04d.png" -r 2 -crf 30 bbb.flv
ffmpeg -y -i "images/test_%%04d.png" -an -r 10  -pix_fmt rgb24 -vf "setpts=2.0*PTS"  -vf "scale=500:-1" -f gif bbb.gif


:end

del *.dll

