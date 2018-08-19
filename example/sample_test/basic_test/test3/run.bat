set PYPATH=C:\Users\vaio6\Anaconda3
set PYPATH=C:\Users\neutral\Anaconda3

set LDM=..\..\..\all_build\x64\Release
copy ..\..\..\..\third_party\bin\*.dll

%LDM%\test3.exe

:pause


del *.dll