set PYPATH=%USERPROFILE%\Anaconda3

set LDM=..\..\..\all_build\x64\Release
copy ..\..\..\..\third_party\bin\*.dll

set progress=1
if "%1"=="0" set progress=0

del /Q images\*.png

%LDM%\TimeSeriesRegression.exe  --header 1 --x 1 --y 1 --csv "annual-changes-in-global-tempera.csv" ^
--learning_rate 1 --opt_type adam ^
--test 0.11 --progress %progress% --plot 5 --tol 0.0001 --early_stopping 0 ^
--epochs 26000 --n_layers 3  --n_rnn_layers 1 --rnn_type lstm ^
--seq_len 24 --minibatch_size 24 --hidden_size 64 --test_mode 0 --normal zscore --support 0 --prophecy 60 --test_mode 0
: > log1.txt

goto end
ffmpeg -y -i "images/test_%%04d.png" -r 2 -crf 30 bbb.mp4
ffmpeg -y -i "images/test_%%04d.png" -r 2 -crf 30 bbb.flv
ffmpeg -y -i "images/test_%%04d.png" -an -r 10  -pix_fmt rgb24 -vf "setpts=2.0*PTS"  -vf "scale=500:-1" -f gif bbb.gif


:end

del *.dll

