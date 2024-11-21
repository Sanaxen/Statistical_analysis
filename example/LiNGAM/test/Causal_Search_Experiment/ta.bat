del /Q Digraph*.png

setlocal enabledelayedexpansion
set num=0

:1

if exist ..\work\lingam.model.update (
	
	copy ..\work\lingam.model.update lingam.model.update.txt /v /y
	call lingam_graph.bat
	del ..\work\lingam.model.update
	echo %num%

	copy Digraph.png Digraph_%num%.png /v /y
	:python sendimg.py
	:call line.bat
	
	del lingam_output.zip
	zip -u -j lingam_output.zip ..\work\lingam.model.B.csv
	zip -u -j lingam_output.zip ..\work\lingam.model.B_pre_sort.csv
	zip -u -j lingam_output.zip ..\work\lingam.model.colnames_id
	zip -u -j lingam_output.zip ..\work\lingam.model.hidden_colnames_id
	zip -u -j lingam_output.zip ..\work\lingam.model.importance_B.csv
	zip -u -j lingam_output.zip ..\work\lingam.model.input.csv
	zip -u -j lingam_output.zip ..\work\lingam.model.input_sample.csv
	zip -u -j lingam_output.zip ..\work\lingam.model.intercept.csv
	zip -u -j lingam_output.zip ..\work\lingam.model.lingam_loss.dat
	zip -u -j lingam_output.zip ..\work\lingam.model.loss
	zip -u -j lingam_output.zip ..\work\lingam.model.modification_input.csv
	zip -u -j lingam_output.zip ..\work\lingam.model.mutual_information.csv
	zip -u -j lingam_output.zip ..\work\lingam.model.replacement
	zip -u -j lingam_output.zip ..\work\lingam.model.residual_error.csv
	zip -u -j lingam_output.zip ..\work\lingam.model.residual_error_independ.csv
	zip -u -j lingam_output.zip ..\work\lingam.model.b_probability.csv
	zip -u -j lingam_output.zip ..\work\comandline_args
	
	set /a num=num+1
	
	if %num%==5 exit
	if %num%=="5" exit
	
)
timeout /t 5


goto 1
endlocal
