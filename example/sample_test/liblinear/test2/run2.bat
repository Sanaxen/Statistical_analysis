set PYPATH=%USERPROFILE%\Anaconda3

set LDM=..\..\..\all_build\x64\Release
copy ..\..\..\..\third_party\bin\*.dll


%LDM%\libliner.exe --csv Breast-Cancer-Wisconsin-Diagnostic.csv --header 1   --y_var "diagnosis" ^
--x_var "radius_mean" --x_var "texture_mean" --x_var "perimeter_mean" --x_var "area_mean" --x_var "smoothness_mean" ^
--x_var "compactness_mean" --x_var "concavity_mean" --x_var "concave points_mean" --x_var "symmetry_mean" ^
--x_var "fractal_dimension_mean" --x_var "radius_se" --x_var "texture_se" --x_var "perimeter_se" --x_var "area_se" ^
--x_var "smoothness_se" --x_var "compactness_se" --x_var "concavity_se" --x_var "concave points_se" --x_var "symmetry_se" ^
--x_var "fractal_dimension_se" --x_var "radius_worst" --x_var "texture_worst" --x_var "perimeter_worst" --x_var "area_worst" ^
--x_var "smoothness_worst" --x_var "compactness_worst" --x_var "concavity_worst" --x_var "concave points_worst" ^
--x_var "symmetry_worst" --x_var "fractal_dimension_worst"



%LDM%\logistic_regression.exe --train 1 --L2 0.000001 --file Breast-Cancer-Wisconsin-Diagnostic.csv_libsvm.format --model model --cv_report cv.txt
%LDM%\logistic_regression.exe --predict 1 --file Breast-Cancer-Wisconsin-Diagnostic.csv_libsvm.format --model model --output out.txt

:pause
del *.dll


