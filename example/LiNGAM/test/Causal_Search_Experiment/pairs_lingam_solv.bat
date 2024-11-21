set bin=..\bin\gpu_version\LiNGAM_cuda.exe
:set bin=..\bin\LiNGAM.exe

del ..\work\causal_multi_histgram.png
del ..\work\Digraph.png
del ..\work\b_probability.png
del ..\work\Causal_effect.png
del ..\work\b_importance.png
del ..\work\fit.png
del ..\work\scatter.png
del ..\work\err_histogram.png

del ..\work\b_probability_barplot.r
del ..\work\Causal_effect.r
del ..\work\fit.r
del ..\work\b_importance.r
del ..\work\scatter.r
del ..\work\scatter2.r
del ..\work\error_hist.r


set learning_rate=0.001
:default=0.1
set distribution_rate=0.1
set unit=20
set layer=3
set epoch=20
set use_gpu=0
set activation=selu
set use_pnl=1
set sampling_iter=600

set csv=pairs\pair%1.csv
:copy pairs_comandline_args ..\work\comandline_args /v /y
copy pairs_comandline_args_lingam ..\work\comandline_args /v /y

:pause
copy %csv% ..\work\tmp_Causal_relationship_search.csv /v /y
copy %csv% ..\work\pair%1.csv /v /y
:pause

if not "%3"=="" (
	set sampling_iter=%3
)

cd ..\work
type comandline_args > comandline_args_tmp_
echo  --activation_fnc %activation% >> comandline_args_tmp_
echo  --learning_rate %learning_rate% >> comandline_args_tmp_
:echo  --use_bootstrap 1 >> comandline_args_tmp_
echo  --distribution_rate %distribution_rate% >> comandline_args_tmp_
echo  --n_unit %unit% >> comandline_args_tmp_
echo  --n_layer %layer% >> comandline_args_tmp_
echo  --n_epoch %epoch% >> comandline_args_tmp_
echo  --use_gpu %use_gpu% >> comandline_args_tmp_
echo  --confounding_factors_sampling %sampling_iter% >> comandline_args_tmp_
echo  --rho 3 >> comandline_args_tmp_
echo  --optimizer rmsprop >> comandline_args_tmp_
echo  --csv pair%1.csv >> comandline_args_tmp_
echo  --minbatch 0 >> comandline_args_tmp_
echo  --confounding_factors_upper2 0.5 >> comandline_args_tmp_
echo  --u1_param 0.001 >> comandline_args_tmp_
:echo  --early_stopping 4000 >> comandline_args_tmp_
echo  --confounding_factors 1 >> comandline_args_tmp_
echo  --nonlinear 1 >> comandline_args_tmp_
echo  --normalize_type 2 >> comandline_args_tmp_
echo  --dropout_rate 0.0 >> comandline_args_tmp_
echo  --use_pnl %use_pnl%  >> comandline_args_tmp_
echo  --random_pattern 1 >> comandline_args_tmp_

echo  --pause 0 >> comandline_args_tmp_

%bin% --@ comandline_args_tmp_
:%bin% --@ comandline_args

cd ..\Causal_Search_Experiment

:if not "%2"=="" (
:	echo %2 > ..\work\lingam.model.update
:)
:pause

call lingam_graph.bat
copy Digraph.png Digraph_%1.png /v /y
copy log.txt log%1.txt /v /y
copy scatter.png scatter_%1.png /v /y
copy scatter2.png scatter2_%1.png /v /y

copy pattern_count.txt pattern_count_%1.txt /v /y

call line.bat Digraph_%1.png
:pause

