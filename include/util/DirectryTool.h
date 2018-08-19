#ifndef _DirectryTool_h
#define _DirectryTool_h
//Copyright (c) 2018, Sanaxn
//All rights reserved.

#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <string>

#include "windows.h"

#include <imagehlp.h>					//MakeSureDirectoryPathExists
#pragma comment(lib, "imagehlp.lib")

#include <shlwapi.h>					//PathIsDirectoryA
#pragma comment(lib, "shlwapi.lib")


class DirectryTool
{
public:
	bool ExistDir(char* dirname)
	{
		return (bool)PathIsDirectoryA(dirname);

	}

	bool MakeDir(char* dirname)
	{
		if ( ExistDir(dirname) )
		{
			return true;
		}
		return (bool)MakeSureDirectoryPathExists(dirname);
	}
};

#endif