set PYPATH=%USERPROFILE%\Anaconda3

set LDM=..\..\..\all_build\x64\Release
copy ..\..\..\..\third_party\bin\*.dll

set progress=1
if "%1"=="0" set progress=0

del /Q images\*.png

%LDM%\TimeSeriesRegression.exe  --header 1 --x 0 --y 1 --y_var 1 --t_var 0 --csv "co2-ppm-mauna-loa-19651980.csv" ^
--learning_rate 1 --opt_type adam ^
--test 0.01 --progress %progress% --plot 5 --tol 0.0001 --early_stopping 1 ^
--epochs 26000 --n_layers 5  --n_rnn_layers 1 --rnn_type lstm --use_cnn 2 ^
--seq_len 36 --minibatch_size 36 --hidden_size 200 --test_mode 0 --normal zscore --prophecy 60
: > log1.txt
goto end



ffmpeg -y -i "images/test_%%04d.png" -r 2 -crf 30 bbb.mp4
ffmpeg -y -i "images/test_%%04d.png" -r 2 -crf 30 bbb.flv
ffmpeg -y -i "images/test_%%04d.png" -an -r 10  -pix_fmt rgb24 -vf "setpts=2.0*PTS"  -vf "scale=500:-1" -f gif bbb.gif


:end

del *.dll

