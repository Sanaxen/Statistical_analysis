set cur=%~dp0
set bin=%cur%\bin\LiNGAM_cuda.exe
copy ..\..\..\all_build\x64\Release_pytorch\LiNGAM_cuda.exe bin /v /y

if not exist ".\bin\rnn6.dll"  (
    echo rnn6.dll not found™ (https://github.com/Sanaxen/cpp_torch/tree/master/cpp_torch/test/rnn6)
) 

:set OMP_NUM_THREADS=5

:set csv=nonlinear_LiNGAM_latest3a.csv
:set csv=nonlinear_LiNGAM_latest3b.csv
:set csv=nonlinear_LiNGAM_latest3c.csv
:set csv=LiNGAM_latest3.csv
:set csv=nonlinear.csv
:set csv=nonlinear2.csv
:set csv=fMRI_sim1.csv
set csv=fMRI_sim2.csv

set learning_rate=0.01
:default=0.1
set distribution_rate=1
set unit=40
set layer=3
set epoch=60
:set activation=leakyrelu
set activation=selu
:set activation=tanh
:set activation=mish
set use_gpu=0
set use_pnl=1

if "%csv%"=="fMRI_sim1.csv" copy fMRI_sim1_comandline_args ..\work\comandline_args /v /y
if "%csv%"=="fMRI_sim2.csv" copy fMRI_sim2_comandline_args ..\work\comandline_args /v /y

if "%csv%"=="nonlinear.csv" copy nonlinear_comandline_args ..\work\comandline_args /v /y
if "%csv%"=="nonlinear2.csv" copy nonlinear2_comandline_args ..\work\comandline_args /v /y
if "%csv%"=="LiNGAM_latest3.csv" copy LiNGAM_latest3_comandline_args ..\work\comandline_args /v /y
if "%csv%"=="nonlinear_LiNGAM_latest3a.csv" copy LiNGAM_latest3_comandline_args ..\work\comandline_args /v /y
if "%csv%"=="nonlinear_LiNGAM_latest3b.csv" copy LiNGAM_latest3_comandline_args ..\work\comandline_args /v /y
if "%csv%"=="nonlinear_LiNGAM_latest3c.csv" copy LiNGAM_latest3_comandline_args ..\work\comandline_args /v /y

:pause
copy ..\%csv% ..\work\tmp_Causal_relationship_search.csv /v /y
copy ..\%csv% ..\work\%csv% /v /y

cd ..\work
type comandline_args > comandline_args_tmp_
echo  --activation_fnc %activation% >> comandline_args_tmp_
echo  --learning_rate %learning_rate% >> comandline_args_tmp_
echo  --use_bootstrap 1 >> comandline_args_tmp_
echo  --distribution_rate %distribution_rate% >> comandline_args_tmp_
echo  --n_unit %unit% >> comandline_args_tmp_
echo  --n_layer %layer% >> comandline_args_tmp_
echo  --n_epoch %epoch% >> comandline_args_tmp_
echo  --use_gpu %use_gpu% >> comandline_args_tmp_
echo  --confounding_factors_sampling 30000 >> comandline_args_tmp_
echo  --rho 3 >> comandline_args_tmp_
echo  --optimizer rmsprop >> comandline_args_tmp_
echo  --csv %csv% >> comandline_args_tmp_
echo  --minbatch 0 >> comandline_args_tmp_
echo  --confounding_factors_upper2 0.05 >> comandline_args_tmp_
echo  --u1_param 0.001 >> comandline_args_tmp_
echo  --use_pnl %use_pnl%  >> comandline_args_tmp_
echo  --random_pattern 0 >> comandline_args_tmp_
echo  --dropout_rate 0.01  >> comandline_args_tmp_
echo  --_Causal_Search_Experiment 1 >>  comandline_args_tmp_
:echo  --use_hsic 1 >>  comandline_args_tmp_

%bin% --@ comandline_args_tmp_
:%bin% --@ comandline_args

cd ..\Causal_Search_Experiment

