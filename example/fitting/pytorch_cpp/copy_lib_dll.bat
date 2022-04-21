set s=D:\torch\cpp_torch\test\rnn6\rnn6\x64\Release

copy %s%\rnn6.dll bin /v /y
copy %s%\rnn6.lib lib /v /y
copy D:\torch\cpp_torch\test\rnn6\tiny_dnn2libtorch_dll.h .\ /v /y

pause
