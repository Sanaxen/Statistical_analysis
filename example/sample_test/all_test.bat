del /s plot_*.*
echo Test > log.txt


cd basic_test
call runall.bat >> ..\log.txt
cd ..

cd fastICA
call runall.bat >> ..\log.txt
cd ..

cd hungarian-algorithm
call runall.bat >> ..\log.txt
cd ..

cd lasso_lib_test
call runall.bat >> ..\log.txt
cd ..

cd LiNGAM
call runall.bat >> ..\log.txt
cd ..

cd PCA
call runall.bat >> ..\log.txt
cd ..

cd nonliner
call runall.bat >> ..\log.txt
cd ..

