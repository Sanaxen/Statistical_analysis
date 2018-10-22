set PYPATH=%USERPROFILE%\Anaconda3

set LDM=..\..\..\all_build\x64\Release

copy ..\..\..\..\third_party\bin\*.dll

set progress=1
if "%1"=="0" set progress=0

del /Q images\*.png
%LDM%\NonLinearRegression.exe  --x 2 --y 3 --csv sample.csv --test 0 --progress %progress%  --plot 5

ffmpeg -y -i "images/test_%%04d.png" -r 2 -crf 30 aaa.mp4
ffmpeg -y -i "images/test_%%04d.png" -r 2 -crf 30 aaa.flv
ffmpeg -y -i "images/test_%%04d.png" -an -r 10  -pix_fmt rgb24 -vf "setpts=2.0*PTS"  -vf "scale=500:-1" -f gif aaa.gif

del *.dll