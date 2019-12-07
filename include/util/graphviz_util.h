namespace graphviz_path_
{
	class app {
	public:
		app() {
			char buf[MAX_PATH + 1];
			if (0 != GetModuleFileNameA(NULL, buf, MAX_PATH)) {// 実行ファイルの完全パスを取得
				char drive[MAX_PATH + 1]
					, dir[MAX_PATH + 1]
					, fname[MAX_PATH + 1]
					, ext[MAX_PATH + 1];
				_splitpath(buf, drive, dir, fname, ext);//パス名を構成要素に分解します
				sprintf(path, "%s%s", drive, dir);
			}
		}
		char path[MAX_PATH+1];
	};

	std::string path_ = "";
	inline void getGraphvizPath()
	{
		if (graphviz_path_::path_ != "") return;

		app mypath;
		std::string path = std::string(mypath.path) + std::string("graphviz_path.txt");

		//FILE* fp_tmp = fopen("debuglog.txt", "w");
		//fprintf(fp_tmp, "%s\n", path.c_str());

		FILE* fp = fopen(path.c_str(), "r");
		if (fp)
		{
			char buf[640];
			fgets(buf, 640, fp);
			char* p = strstr(buf, "\n");
			if (p) *p = '\0';
			graphviz_path_::path_ = std::string(buf);
			fclose(fp);
		}
	}
}
