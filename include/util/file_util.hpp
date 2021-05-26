#ifndef _FILE_UTIL_HPP
#define _FILE_UTIL_HPP
//Copyright (c) 2018, Sanaxn
//All rights reserved.

inline FILE* fopen_util(const char* filename, const char* mode)
{
	FILE* fp = NULL;
	
	for ( int k = 0; k < 1000; k++ )
	{
		fp = fopen(filename, mode);
		if ( fp ) break;
		printf("open [%s] waiting!! %d/%d\n", filename, k+1, 1000);
		Sleep(100);
	}

	return fp;
}

inline void copyfile(const char *from_file_name, const char *to_file_name)
{
	ifstream is(from_file_name, ios::in | ios::binary);
	ofstream os(to_file_name, ios::out | ios::binary);
	os << is.rdbuf();
}

#endif
