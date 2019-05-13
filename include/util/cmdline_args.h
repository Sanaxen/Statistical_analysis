#ifndef _CMD_LINES_ARGS_H

#define _CMD_LINES_ARGS_H

inline int commandline_args(int* argc_, char*** argv_)
{
	int argc = *argc_;
	char** argv = *argv_;

	std::string resp = "";
	for (int count = 1; count + 1 < argc; count += 2)
	{
		std::string argname(argv[count]);
		if (argname == "--@") {
			resp = std::string(argv[count + 1]);
			continue;
		}
	}
	if (resp != "")
	{
		std::ifstream ifs(resp);
		std::string lines((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
		std::cout << lines << std::endl;

		if (strstr(lines.c_str(), "--"))
		{ 
			std::string tmp = " " + lines;
			lines = tmp;
		}
		else
		{
			return -1;
		}
		char* commandline = new char[lines.length() + 1];
		strcpy(commandline, lines.c_str());

		std::vector<std::string> args;
		args.push_back(argv[0]);

		char* p = commandline;
		while (isspace(*p) || *p == '\n') p++;
		p--;
		while (*p != '\0')
		{
			char* q = strstr(p, " --");
			q++;
			char* s = strchr(q, ' ');
			*s = '\0';
			s++;
			args.push_back(q);

			while (isspace(*s) ||*s == '\n') s++;
			q = strstr(s, " --");
			if (q)
			{
				while (isspace(*q) || *q == '\n') q--;
				q++;
				*q = '\0';
				args.push_back(s);
				*q = ' ';
			}
			else
			{
				char* q = strchr(s, '\0');
				q--;
				while (isspace(*q) || *q == '\n')q--;
				q++;
				*q = '\0';
				args.push_back(s);
				break;
			}

			p = q;
		}
		argc = args.size();
		argv = new char*[argc];
		for (int i = 0; i < argc; i++)
		{
			argv[i] = new char[args[i].length() + 1];
			strcpy(argv[i], args[i].c_str());
		}

		*argv_ = argv;
		*argc_ = argc;
		return 0;
	}
	return 1;
}
#endif
