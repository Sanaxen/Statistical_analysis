set PYPATH=%USERPROFILE%\Anaconda3

set LDM=..\..\..\all_build\x64\Release

copy ..\..\..\..\third_party\bin\*.dll

set progress=1
if "%1"=="0" set progress=0

del /Q images\*.png
%LDM%\NonLinearRegression.exe  --x 1 --y 1 --csv sample2.csv --test 0 --progress %progress%  --plot 10 --epochs 200
:--minibatch_size 30 --learning_rate 0.1 --n_layers 4 --input_unit 64

ffmpeg -y -i "images/test_%%04d.png" -r 2 -crf 30 ccc.mp4
ffmpeg -y -i "images/test_%%04d.png" -r 2 -crf 30 ccc.flv
ffmpeg -y -i "images/test_%%04d.png" -an -r 10  -pix_fmt rgb24 -vf "setpts=2.0*PTS"  -vf "scale=500:-1" -f gif ccc.gif

del *.dll
