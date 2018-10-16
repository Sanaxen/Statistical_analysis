set PYPATH=%USERPROFILE%\Anaconda3


set LDM=..\..\..\all_build\x64\Release
copy ..\..\..\..\third_party\bin\*.dll

%LDM%\preTestLiNGAM.exe

:pause
del *.dll