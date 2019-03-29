set PYPATH=%USERPROFILE%\Anaconda3


set LDM=..\..\..\all_build\x64\Release
copy ..\..\..\..\third_party\bin\*.dll


del /Q images\*.png
%LDM%\TimeSeriesRegression.exe --header 1 --x 1 --y 1 ^
--y_var "�f�[�^" --x_var "�x���t���O2" --t_var 0 ^
--csv sample.csv --test 0.2 --progress 1 --plot 50 --learning_rate 1 --tol 0.0001 ^
--seq_len 200 --minibatch_size 200 --hidden_size 200 ^
--n_layers 2 --n_rnn_layers 1 --read_max 1000 --normal minmax --test_mode 0 --support 0
goto end


:ffmpeg -y -i "images/test_%%04d.png" -r 2 -crf 30 bbb.mp4
:ffmpeg -y -i "images/test_%%04d.png" -r 2 -crf 30 bbb.flv
:ffmpeg -y -i "images/test_%%04d.png" -an -r 10  -pix_fmt rgb24 -vf "setpts=2.0*PTS"  -vf "scale=500:-1" -f gif bbb.gif

:end
del *.dll