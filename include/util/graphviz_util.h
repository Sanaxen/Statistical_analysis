namespace graphviz_path_
{
	std::string path_ = "";
	inline void getGraphvizPath()
	{
		if (graphviz_path_::path_ != "") return;
		FILE* fp = fopen("graphviz_path.txt", "r");
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
