#ifndef _FILECHECK_HPP
#define _FILECHECK_HPP

#include <Windows.h>

class FileExitsCheck
{
	WIN32_FIND_DATAA FindFileData;
	HANDLE hFind;
public:

	FileExitsCheck() { hFind = NULL; }

	inline bool isExist(std::string& filename)
	{
		hFind = FindFirstFileA(filename.c_str(), &FindFileData);
		if (hFind == INVALID_HANDLE_VALUE) 
		{
			// 存在しない場合
			return false;
		}
		// 存在する場合
		FindClose(hFind);
		return true;
	}

};
#endif
