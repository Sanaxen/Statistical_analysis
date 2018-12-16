set PYPATH=%USERPROFILE%\Anaconda3

set LDM=..\..\..\all_build\x64\Release

copy ..\..\..\..\third_party\bin\*.dll

set progress=1
if "%1"=="0" set progress=0

del /Q images\*.png
:%LDM%\NonLinearRegression.exe  --header 1 --x 2 --y 3 --x_var "x1" --x_var "x2" --y_var "y1" --y_var "y2" --y_var "y3" --csv sample3.csv --test 0 --progress %progress%  --plot 5
:%LDM%\NonLinearRegression.exe  --header 1 --x 2 --y 3 --x_var "x1" --x_var "x2" --csv sample3.csv --test 0 --progress %progress%  --plot 5
:%LDM%\NonLinearRegression.exe  --header 1 --x 2 --y 3  --y_var "y1" --y_var "y2" --y_var "y3" --csv sample3.csv --test 0 --progress %progress%  --plot 5

%LDM%\NonLinearRegression.exe  --x 2 --y 3 --csv sample.csv --test 0.5 --tol 0.005 --progress %progress%  --plot 5 --epochs 6000  --minibatch_size 64
goto end

%LDM%\NonLinearRegression.exe  --x 2 --y 3 --csv sample.csv --test 0.0 --tol 0.05 --progress %progress%  --plot 5 --epochs 6000

ffmpeg -y -i "images/test_%%04d.png" -r 2 -crf 30 aaa.mp4
ffmpeg -y -i "images/test_%%04d.png" -r 2 -crf 30 aaa.flv
ffmpeg -y -i "images/test_%%04d.png" -an -r 10  -pix_fmt rgb24 -vf "setpts=2.0*PTS"  -vf "scale=500:-1" -f gif aaa.gif

:end

del *.dll

