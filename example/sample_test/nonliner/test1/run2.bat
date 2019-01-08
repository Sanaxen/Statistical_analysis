set PYPATH=%USERPROFILE%\Anaconda3

set LDM=..\..\..\all_build\x64\Release

copy ..\..\..\..\third_party\bin\*.dll

set progress=1
if "%1"=="0" set progress=0

del /Q images\*.png
%LDM%\NonLinearRegression.exe  --x 1 --y 1 --csv sample2.csv --test 0 ^
--progress %progress%  --plot 10 --epochs 600 ^
--minibatch_size 60 --learning_rate 1 --n_layers 4 --input_unit 64


del *.dll
