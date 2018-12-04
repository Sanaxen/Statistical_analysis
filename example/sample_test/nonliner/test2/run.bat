set PYPATH=%USERPROFILE%\Anaconda3


set LDM=..\..\..\all_build\x64\Release
copy ..\..\..\..\third_party\bin\*.dll

set progress=1
if "%1"=="0" set progress=0

del /Q images\*.png
:%LDM%\TimeSeriesRegression.exe --header 1  --x 2 --y 3 --x_var "t1" --x_var "t2" --y_var "y1" --y_var "y2" --y_var "y3" --csv sample2.csv --test 0.5 --progress %progress% --plot 50 --tol 0.014

%LDM%\TimeSeriesRegression.exe  --x 2 --y 3 --csv sample.csv --test 0.5 --progress %progress% --plot 5 --tol 0.05 --epochs 600 --n_layers 5 --seq_len 64 --input_unit 32 --minibatch_size 64

goto end
:%LDM%\TimeSeriesRegression.exe  --x 2 --y 3 --csv sample.csv --test 0.5 --progress %progress% --plot 5 --tol 0.01 --epochs 600 --n_layers 8 --seq_len 20 --input_unit 64
%LDM%\TimeSeriesRegression.exe  --x 2 --y 3 --csv sample.csv --test 0.5 --progress %progress% --plot 5 --tol 0.01 --epochs 600 --n_layers 5 --seq_len 20 --input_unit 64
:%LDM%\TimeSeriesRegression.exe  --x 2 --y 3 --csv sample.csv --test 0.5 --progress %progress% --plot 5 --tol 0.01 --epochs 300

: --input_unit 64 --learning_rate 0.001 --minibatch_size 48  --n_layers 5  --n_rnn_layers 2


ffmpeg -y -i "images/test_%%04d.png" -r 2 -crf 30 bbb.mp4
ffmpeg -y -i "images/test_%%04d.png" -r 2 -crf 30 bbb.flv
ffmpeg -y -i "images/test_%%04d.png" -an -r 10  -pix_fmt rgb24 -vf "setpts=2.0*PTS"  -vf "scale=500:-1" -f gif bbb.gif


del *.dll

:end
