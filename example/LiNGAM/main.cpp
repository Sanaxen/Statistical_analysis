//#define USE_MKL
#define _cublas_Init_def
#include "../../include/statistical/LiNGAM.h"

#ifdef USE_GNUPLOT
#include "../../include/util/plot.h"

#endif
#include "../../include/util/cmdline_args.h"

vector<string> split(string str, string separator) 
{
	if (separator == "") return { str };
	vector<string> result;
	string tstr = str + separator;
	long l = tstr.length(), sl = separator.length();
	string::size_type pos = 0, prev = 0;

	for (; pos < l && (pos = tstr.find(separator, pos)) != string::npos; prev = (pos += sl)) {
		result.emplace_back(tstr, prev, pos - prev);
	}
	return result;
}
bool prior_knowledge(const char* filename, std::vector<std::string>& header_names, std::vector<int>& m)
{
	printf("prior_knowledge[%s]\n", filename);
	FILE* fp = fopen(filename, "r");
	if (!fp) return false;

	char buf[256];
	while (fgets(buf, 256, fp) != NULL)
	{
		char* p = strchr(buf, '\n');
		if (p) *p = '\0';

		if (buf[0] == '#') continue;
		if (buf[0] == '\0') continue;

		printf("---[%s]------\n", buf);
		std::string str(buf);
		auto tokens1 = split(str, "<-");
		if (tokens1.size() == 2)
		{
			for (auto t : tokens1) 
			{
				for (int i = 0; i < header_names.size(); i++)
				{
					//std::cout << t << " " << header_names[i] << std::endl;
					if (t == header_names[i])
					{
						m.push_back(-(i + 1));
						break;
					}else
					if ("\""+t +"\"" == header_names[i])
					{
						m.push_back(-(i + 1));
						break;
					}

					if (t == "_" )
					{
						int id = abs(m[m.size() - 1])-1;
						m.pop_back();
						for (int j = 0; j < header_names.size(); j++)
						{
							if (j == id) continue;
							m.push_back(-(id + 1));
							m.push_back(-(j + 1));
						}
						break;
					}
				}
			}
		}
		//printf("%d\n", m.size());
		if (m.size() % 2)
		{
			printf("ERROR:prior_knowledge\n");
		}
		auto tokens2 = split(str, "->");

		if (tokens2.size() == 2)
		{
			for (auto t : tokens2)
			{
				for (int i = 0; i < header_names.size(); i++)
				{
					//std::cout << t << " " << header_names[i] << std::endl;
					if (t == header_names[i])
					{
						m.push_back(i + 1);
						break;
					}
					else
					if ("\"" + t + "\"" == header_names[i])
					{
						m.push_back(i + 1);
						break;
					}
					if (t == "_" )
					{
						int id = abs(m[m.size() - 1]) - 1;
						m.pop_back();
						for (int j = 0; j < header_names.size(); j++)
						{
							if (j == id) continue;
							m.push_back((id + 1));
							m.push_back((j + 1));
						}
						break;
					}
				}
			}
		}
		//printf("%d\n", m.size());
		if (m.size() % 2)
		{
			printf("ERROR:prior_knowledge\n");
		}
	}

	for (int i = 0; i < m.size()/2; i++)
	{
		if (m[2 * i] < 0)
		{
			printf("X %d<-%d\n", abs(m[2 * i]) - 1, abs(m[2 * i + 1]) - 1);
		}
		if (m[2 * i] > 0)
		{
			printf("O %d->%d\n", abs(m[2 * i]) - 1, abs(m[2 * i + 1]) - 1);
		}
	}
	fclose(fp);

	if (m.size() % 2)
	{
		printf("ERROR:prior_knowledge\n");
	}
	else
	{
		printf("prior_knowledge[%s] success\n", filename);
	}
	return true;
}

//https://qiita.com/m__k/items/bd87c063a7496897ba7c
//LiNGAMモデルの推定方法について
int main(int argc, char** argv)
{
	SigHandler_lingam(0);
	signal(SIGINT, SigHandler_lingam);
	signal(SIGTERM, SigHandler_lingam);
	signal(SIGBREAK, SigHandler_lingam);
	signal(SIGABRT, SigHandler_lingam);

	int resp = commandline_args(&argc, &argv);
	if (resp == -1)
	{
		printf("command line error.\n");
		return -1;
	}
	std::mt19937 mt(1234);

	std::string csvfile("sample.csv");

	int start_col = 0;
	bool header = false;
	for (int count = 1; count + 1 < argc; count += 2) {
		std::string argname(argv[count]);
		if (argname == "--csv") {
			csvfile = std::string(argv[count + 1]);
		}
		if (argname == "--header") {
			header = (atoi(argv[count + 1]) != 0) ? true : false;
		}
		if (argname == "--col") {
			start_col = atoi(argv[count + 1]);
		}
	}


	FILE* fp = fopen(csvfile.c_str(), "r");
	if (fp == NULL)
	{
		Matrix<dnn_double> x, y, z, w;
		CSVReader csv1("x.csv", ',', false);
		CSVReader csv2("y.csv", ',', false);
		CSVReader csv3("z.csv", ',', false);
		CSVReader csv6("w.csv", ',', false);
		x = csv1.toMat().transpose();
		y = csv2.toMat().transpose();
		z = csv3.toMat().transpose();
		w = csv6.toMat().transpose();
		Matrix<dnn_double>xs(x.m, 4);

		for (int i = 0; i < x.m; i++)
		{
			xs(i, 0) = x(i, 0);
			xs(i, 1) = y(i, 0);
			xs(i, 2) = z(i, 0);
			xs(i, 3) = w(i, 0);
		}
		xs.print_csv("sample.csv");
		header = false;
	}

	//CSVReader csv1(csvfile, ',', header);
	//Matrix<dnn_double> xs = csv1.toMat();
	//xs = csv1.toMat_removeEmptyRow();
	//if (start_col)
	//{
	//	for (int i = 0; i < start_col; i++)
	//	{
	//		xs = xs.removeCol(0);
	//	}
	//}


	Lingam LiNGAM;

	size_t max_ica_iteration = MAX_ITERATIONS;
	double ica_tolerance = TOLERANCE;
	double lasso = 0.0;

	double cor_range[2] = { 0,0 };
	double min_cor_delete = -1;
	double min_delete = -1;
	double min_delete_srt = -1;
	bool error_distr = false;
	int error_distr_size[2] = { 1,1 };
	bool capture = true;
	bool sideways = false;
	int diaglam_size = 20;
	char* output_diaglam_type = "png";
	std::vector<std::string> x_var;
	std::vector<std::string> y_var;
	std::vector<std::string> c_var;
	bool ignore_constant_value_columns = true;
	double lasso_tol = TOLERANCE;
	int lasso_itr_max = 1000000;
	bool confounding_factors = false;
	bool view_confounding_factors = false;
	int confounding_factors_sampling = 7000;
	double mutual_information_cut = 0.0;
	bool mutual_information_values = true;
	double distribution_rate = 0.1;
	double temperature_alp = 0.95;
	std::string prior_knowledge_file = "";
	double prior_knowledge_rate = 1.0;
	double rho = 3.0;
	int early_stopping = 0;
	int parameter_search = 0;
	double confounding_factors_upper = 0.9;
	int bins = 30;
	int normalize_type = 0;
	bool use_intercept = false;
	bool loss_data_load = true;
	int cluster = -1;

	int pause = 0;
	std::string load_model = "";

	for (int count = 1; count + 1 < argc; count += 2) {
		std::string argname(argv[count]);
		if (argname == "--csv") {
			continue;
		}
		else
			if (argname == "--header") {
				continue;
			}
			else
				if (argname == "--col") {
					continue;
				}
				else if (argname == "--iter") {
					max_ica_iteration = atoi(argv[count + 1]);
				}
				else if (argname == "--tol") {
					ica_tolerance = atof(argv[count + 1]);
				}
				else if (argname == "--lasso") {
					lasso = atof(argv[count + 1]);
				}
				else if (argname == "--sideways") {
					sideways = atoi(argv[count + 1]) != 0 ? true : false;
				}
				else if (argname == "--output_diaglam_type") {
					output_diaglam_type = argv[count + 1];
				}
				else if (argname == "--diaglam_size") {
					diaglam_size = atoi(argv[count + 1]);
				}
				else if (argname == "--x_var") {
					x_var.push_back(argv[count + 1]);
				}
				else if (argname == "--y_var") {
					y_var.push_back(argv[count + 1]);
				}
				else if (argname == "--c_var") {
					c_var.push_back(argv[count + 1]);
				}
				else if (argname == "--cluster") {
					cluster = atoi(argv[count + 1]);
				}
				else if (argname == "--error_distr") {
					error_distr = atoi(argv[count + 1]) != 0 ? true : false;
				}
				else if (argname == "--capture") {
					capture = atoi(argv[count + 1]) == 0 ? false : true;
				}
				else if (argname == "--error_distr_size") {
					sscanf(argv[count + 1], "%d,%d", error_distr_size, error_distr_size + 1);
				}
				else if (argname == "--min_cor_delete") {
					min_cor_delete = atof(argv[count + 1]);
				}
				else if (argname == "--min_delete") {
					min_delete = atof(argv[count + 1]);
				}
				else if (argname == "--cor_range_d") {
					cor_range[0] = atof(argv[count + 1]);
				}
				else if (argname == "--cor_range_u") {
					cor_range[1] = atof(argv[count + 1]);
				}
				else if (argname == "--ignore_constant_value_columns")
				{
					ignore_constant_value_columns = atoi(argv[count + 1]) == 0 ? false : true;
				}
				else if (argname == "--lasso_tol")
				{
					lasso_tol = atof(argv[count + 1]);
				}
				else if (argname == "--lasso_itr_max")
				{
					lasso_itr_max = atoi(argv[count + 1]);
				}
				else if (argname == "--confounding_factors")
				{
					confounding_factors = atoi(argv[count + 1]) == 0 ? false : true;
				}
				else if (argname == "--confounding_factors_sampling")
				{
					confounding_factors_sampling = atoi(argv[count + 1]);
				}
				else if (argname == "--mutual_information_cut")
				{
					mutual_information_cut = atof(argv[count + 1]);
				}
				else if (argname == "--mutual_information_values") {
					mutual_information_values = atoi(argv[count + 1]) == 0 ? false : true;
				}
				else if (argname == "--load_model") {
					load_model = argv[count + 1];
				}
				else if (argname == "--distribution_rate") {
					distribution_rate = atof(argv[count + 1]);
				}
				else if (argname == "--temperature_alp") {
					temperature_alp = atof(argv[count + 1]);
				}
				else if (argname == "--prior_knowledge") {
					prior_knowledge_file = argv[count + 1];
				}
				else if (argname == "--prior_knowledge_rate") {
					prior_knowledge_rate = atof(argv[count + 1]);
				}
				else if (argname == "--rho") {
					rho = atof(argv[count + 1]);
				}
				else if (argname == "--bins") {
					bins = atoi(argv[count + 1]);
				}
				else if (argname == "--early_stopping") {
					early_stopping = atoi(argv[count + 1]);
				}
				else if (argname == "--confounding_factors_upper") {
					confounding_factors_upper = atof(argv[count + 1]);
				}
				else if (argname == "--view_confounding_factors") {
					 view_confounding_factors = atoi(argv[count + 1]) == 0 ? false : true;
				}
				else if (argname == "--pause") {
					pause = atoi(argv[count + 1]);
				}
				else if (argname == "--normalize_type") {
					normalize_type = atoi(argv[count + 1]);
				}
				else if (argname == "--use_intercept") {
					use_intercept = atoi(argv[count + 1]) == 0 ? false : true;
				}
				else if (argname == "--min_delete_srt") {
					min_delete_srt = atoi(argv[count + 1]);
				}
				else if (argname == "--loss_data_load") {
					loss_data_load = atoi(argv[count + 1]) == 0 ? false : true;
				}
				///
				//
				else {
					std::cerr << "Invalid parameter specified - \"" << argname << "\""
						<< std::endl;
					return -1;
				}

	}
#if 0
	std::vector<std::string> header_names;
	header_names.resize(xs.n);
	if (header && csv1.getHeader().size() > 0)
	{
		for (int i = 0; i < xs.n; i++)
		{
			header_names[i] = csv1.getHeader(i + start_col);
		}
	}
	else
	{
		for (int i = 0; i < xs.n; i++)
		{
			char buf[32];
			sprintf(buf, "%d", i);
			header_names[i] = buf;
		}
	}

	std::vector<int> x_var_idx;
	int y_var_idx = -1;

	if (x_var.size())
	{
		for (int i = 0; i < x_var.size(); i++)
		{
			for (int j = 0; j < header_names.size(); j++)
			{
				if (x_var[i] == header_names[j])
				{
					x_var_idx.push_back(j);
				}
				else
				{
					char buf[32];
					sprintf(buf, "%d", j);
					if (x_var[i] == std::string(buf))
					{
						x_var_idx.push_back(j);
					}
					sprintf(buf, "\"%d\"", j);
					if (x_var[i] == std::string(buf))
					{
						x_var_idx.push_back(j);
					}
				}
			}
		}
		if (x_var_idx.size() == 0)
		{
			for (int i = 0; i < x_var.size(); i++)
			{
				x_var_idx.push_back(atoi(x_var[i].c_str()));
			}
		}
		if (x_var_idx.size() != x_var.size())
		{
			printf("--x_var ERROR\n");
		}
	}
#else
	CSVReader csv1(csvfile, ',', header);
	Matrix<dnn_double> T = csv1.toMat();
	T = csv1.toMat_removeEmptyRow();
	if (start_col)
	{
		for (int i = 0; i < start_col; i++)
		{
			T = T.removeCol(0);
		}
	}

	std::vector<std::string> header_names;
	header_names.resize(T.n);
	if (header && csv1.getHeader().size() > 0)
	{
		for (int i = 0; i < T.n; i++)
		{
			header_names[i] = csv1.getHeader(i + start_col);
		}
	}
	else
	{
		for (int i = 0; i < T.n; i++)
		{
			char buf[32];
			sprintf(buf, "%d", i);
			header_names[i] = buf;
		}
	}
	csv1.clear();

	std::vector<int> x_var_idx;
	std::vector<int> y_var_idx;
	std::vector<int> c_var_idx;

	if (x_var.size())
	{
		for (int i = 0; i < x_var.size(); i++)
		{
			for (int j = 0; j < header_names.size(); j++)
			{
				if (x_var[i] == header_names[j])
				{
					x_var_idx.push_back(j);
				}
				else if ("\"" + x_var[i] + "\"" == header_names[j])
				{
					x_var_idx.push_back(j);
				}
				else if ("\"" + header_names[j] + "\"" == x_var[i])
				{
					x_var_idx.push_back(j);
				}
				else
				{
					char buf[32];
					sprintf(buf, "%d", j);
					if (x_var[i] == std::string(buf))
					{
						x_var_idx.push_back(j);
					}
					sprintf(buf, "\"%d\"", j);
					if (x_var[i] == std::string(buf))
					{
						x_var_idx.push_back(j);
					}
				}
			}
		}
		if (x_var_idx.size() == 0)
		{
			for (int i = 0; i < x_var.size(); i++)
			{
				x_var_idx.push_back(atoi(x_var[i].c_str()));
			}
		}
		if (x_var_idx.size() != x_var.size())
		{
			printf("--x_var ERROR\n");
			return -1;
		}
	}
	if (y_var.size())
	{
		for (int i = 0; i < y_var.size(); i++)
		{
			for (int j = 0; j < header_names.size(); j++)
			{
				if (y_var[i] == header_names[j])
				{
					y_var_idx.push_back(j);
				}
				else if ("\"" + y_var[i] + "\"" == header_names[j])
				{
					y_var_idx.push_back(j);
				}
				else if ("\"" + header_names[j] + "\"" == y_var[i])
				{
					y_var_idx.push_back(j);
				}
				else
				{
					char buf[32];
					sprintf(buf, "%d", j);
					if (y_var[i] == std::string(buf))
					{
						y_var_idx.push_back(j);
					}
					sprintf(buf, "\"%d\"", j);
					if (y_var[i] == std::string(buf))
					{
						y_var_idx.push_back(j);
					}
				}
			}
		}
		if (y_var_idx.size() != y_var.size())
		{
			printf("--y_var ERROR\n");
			return -1;
		}
	}

	if (c_var.size())
	{
		for (int i = 0; i < c_var.size(); i++)
		{
			for (int j = 0; j < header_names.size(); j++)
			{
				if (c_var[i] == header_names[j])
				{
					c_var_idx.push_back(j);
				}
				else if ("\"" + c_var[i] + "\"" == header_names[j])
				{
					c_var_idx.push_back(j);
				}
				else if ("\"" + header_names[j] + "\"" == c_var[i])
				{
					c_var_idx.push_back(j);
				}
				else
				{
					char buf[32];
					sprintf(buf, "%d", j);
					if (c_var[i] == std::string(buf))
					{
						c_var_idx.push_back(j);
					}
					sprintf(buf, "\"%d\"", j);
					if (c_var[i] == std::string(buf))
					{
						c_var_idx.push_back(j);
					}
				}
			}
		}
		if (c_var_idx.size() != c_var.size())
		{
			printf("--c_var ERROR\n");
			return -1;
		}
	}
	printf("c_var:%d\n", c_var.size());

	if (x_var.size() == 0 )
	{
		for (int i = 0; i < T.n; i++)
		{
			char buf[32];
			sprintf(buf, "\"%d\"", i);
			x_var.push_back(buf);
			x_var_idx.push_back(i);
		}
	}

	for (int i = 0; i < x_var.size(); i++)
	{
		printf("x_var:%s %d\n", x_var[i].c_str(), x_var_idx[i]);
	}
	for (int i = 0; i < y_var.size(); i++)
	{
		printf("y_var:%s %d\n", y_var[i].c_str(), y_var_idx[i]);
	}
	for (int i = 0; i < c_var.size(); i++)
	{
		printf("c_var:%s %d\n", c_var[i].c_str(), c_var_idx[i]);
	}

	std::vector<std::string> headers_tmp;
	std::vector<std::string> error_cols;

	std::vector<int> class_index;
	Matrix<dnn_double> col_;
	if( c_var.size() > 0)
	{
		col_ = T.Col(c_var_idx[0]);
		for (int k = 0; k < col_.m; k++)
		{
			if (col_(k, 0) == cluster)
			{
				class_index.push_back(k);
			}
		}
		if (class_index.size() == 0)
		{
			printf("ERROR:cluster value.\n");
			return -1;
		}
	}

	int index_start = -1;
	Matrix<dnn_double> col;
	for (int i = 0; i < x_var.size(); i++)
	{
		col = T.Col(x_var_idx[i]);

		if (class_index.size())
		{
			std::vector<double> tmp;
			for (int k = 0; k < class_index.size(); k++)
			{
				tmp.push_back(col(class_index[k], 0));
			}
			col = Matrix<dnn_double>(tmp);
		}

		//printf("[%s]Max-Min:%f\n", header_names[x_var_idx[i]].c_str(), fabs(col.Max() - col.Min()));
		if (fabs(col.Max() - col.Min()) < 1.0e-6)
		{
			if (ignore_constant_value_columns) continue;
			error_cols.push_back(header_names[x_var_idx[i]]);
			col = col + col.RandMT(mt)*0.001;
			break;
		}else
		{
			headers_tmp.push_back(header_names[x_var_idx[i]]);
			index_start = i;
			break;
		}
	}

	if (index_start < 0)
	{
		printf("ERROR:to many constant value columns.\n");
		return -1;
	}
	Matrix<dnn_double> xs = col;
	for (int i = index_start + 1; i < x_var.size(); i++)
	{
		col = T.Col(x_var_idx[i]);
		if (class_index.size())
		{
			std::vector<double> tmp;
			for (int k = 0; k < class_index.size(); k++)
			{
				tmp.push_back(col(class_index[k], 0));
			}
			col = Matrix<dnn_double>(tmp);
		}
		//printf("[%s]Max-Min:%f\n", header_names[x_var_idx[i]].c_str(), fabs(col.Max() - col.Min()));
		if (fabs(col.Max() - col.Min()) < 1.0e-6)
		{
			if (ignore_constant_value_columns) continue;
			error_cols.push_back(header_names[x_var_idx[i]]);
			col = col + col.RandMT(mt)*0.001;
		}
		xs = xs.appendCol(col);
		headers_tmp.push_back(header_names[x_var_idx[i]]);
	}
	for (int i = 0; i < y_var.size(); i++)
	{
		bool dup = false;
		for (int k = 0; k < x_var.size(); k++)
		{
			if (x_var_idx[k] == y_var_idx[i])
			{
				dup = true;
			}
		}
		if (dup) continue;
		col = T.Col(y_var_idx[i]);
		if (class_index.size())
		{
			std::vector<double> tmp;
			for (int k = 0; k < class_index.size(); k++)
			{
				tmp.push_back(col(class_index[k], 0));
			}
			col = Matrix<dnn_double>(tmp);
		}
		//printf("[%s]Max-Min:%f\n", header_names[y_var_idx[i]].c_str(),fabs(col.Max() - col.Min()));
		if (fabs(col.Max() - col.Min()) < 1.0e-6)
		{
			if (ignore_constant_value_columns) continue;
			error_cols.push_back(header_names[y_var_idx[i]]);
			col = col + col.RandMT(mt)*0.001;
		}
		xs = xs.appendCol(col);
		headers_tmp.push_back(header_names[y_var_idx[i]]);
	}
	std::vector<std::string> header_names_org = header_names;
	header_names = headers_tmp;
#endif

	{
		FILE* fp = fopen("select_variables.dat", "w");
		if (fp)fprintf(fp, "%d\n", y_var.size());
		if (y_var.size())
		{
			if (fp)fprintf(fp, "%d,%s\n", y_var_idx[0], header_names[y_var_idx[0]].c_str());
			for (int i = 0; i < y_var_idx.size(); i++)
			{
				if (fp)fprintf(fp, "%d,%s\n", y_var_idx[i], header_names[y_var_idx[i]].c_str());
			}
		}
		for (int i = 0; i < x_var_idx.size(); i++)
		{
			if (fp)fprintf(fp, "%d,%s\n", x_var_idx[i], header_names[x_var_idx[i]].c_str());
		}
		if (fp)fclose(fp);

		fp = fopen("select_variables2.dat", "w");
		if (fp)fprintf(fp, "%d\n", c_var.size());
		if (c_var.size())
		{
			if (fp)fprintf(fp, "%d,%s\n", c_var_idx[0], header_names[c_var_idx[0]].c_str());
			for (int i = 0; i < c_var_idx.size(); i++)
			{
				if (fp)fprintf(fp, "%d,%s\n", c_var_idx[i], header_names[c_var_idx[i]].c_str());
			}
		}
		if (fp)fclose(fp);
	}

	{
		FILE* fp = fopen("error_cols.txt", "r");
		if (fp)
		{
			remove("error_cols.txt");
		}
	}
	if (error_cols.size())
	{
		FILE* fp = fopen("error_cols.txt", "w");
		if (fp)
		{
			for (int i = 0; i < error_cols.size(); i++)
			{
				fprintf(fp, "%s\n", error_cols[i].c_str());
			}
			fclose(fp);
		}
	}

	std::vector<std::string> shapiro_wilk_values_input;
	std::vector<int> shapiro_wilk_test(xs.n, 0);
	Shapiro_wilk_test(xs, header_names, shapiro_wilk_test, shapiro_wilk_values_input);
	{
#ifdef USE_GNUPLOT
		gnuPlot plot1(std::string(GNUPLOT_PATH), 3);
		if (capture)
		{
			plot1.set_capture(error_distr_size, std::string("input_histgram.png"));
		}
		plot1.multi_histgram(std::string("input_histgram.png"), xs, header_names, shapiro_wilk_test);
		plot1.draw();
#endif
	}

	//use_intercept = true;
	if (normalize_type == 1)
	{
		xs = xs / Abs(xs).Max();
	}
	if (normalize_type == 2)
	{
		auto& means = xs.Mean();
#pragma omp parallel for
		for (int i = 0; i < xs.m; i++)
		{
			for (int j = 0; j < xs.n; j++)
			{
				xs(i, j) = xs(i, j) - means(0, j);
			}
		}
	}

	LiNGAM.set(xs.n, mt);
	LiNGAM.mutual_information_values = mutual_information_values;
	LiNGAM.confounding_factors = confounding_factors? 1: 0;
	LiNGAM.confounding_factors_upper = confounding_factors_upper;
	LiNGAM.bins = bins;
	LiNGAM.use_intercept = use_intercept;

	//MutualInformation I(xs.Col(0), xs.Col(1), 30);
	//double tmp = I.Information();
	//printf("MI=%f\n", tmp);
	//fflush(stdout);
	//exit(0);

	//Matrix<double> x(10000, 7);
	//Matrix<double> y(10000, 7);
	//Matrix<double> z(10000, 7);
	//gg_random gg1(2, 1, -3);
	//gg_random gg2(3, 1, -1);
	//gg_random gg3(4, 1, 3);
	//gg_random gg4(5, 1, 5);
	//gg_random gg5(15, 1, 7);
	//gg_random gg6(25, 1, 10);
	//gg_random gg7(55, 1, 13);

	//gg_random gg8(1, 2,  -3);
	//gg_random gg9(1, 3, -1);
	//gg_random gg10(1, 4,  3);
	//gg_random gg11(1, 5, 5);
	//gg_random gg12(1, 15, 7);
	//gg_random gg13(1, 25, 10);
	//gg_random gg14(1, 55, 13);

	//gg_random gg15(10, 45, -3);
	//gg_random gg16(20, 45, -1);
	//gg_random gg17(40, 45, 3);
	//gg_random gg18(45, 10, 5);
	//gg_random gg19(45, 20, 7);
	//gg_random gg20(45, 40, 10);
	//for (int i = 0; i < 10000; i++)
	//{
	//	x(i, 0) = gg1.rand();
	//	x(i, 1) = gg2.rand();
	//	x(i, 2) = gg3.rand();
	//	x(i, 3) = gg4.rand();
	//	x(i, 4) = gg5.rand();
	//	x(i, 5) = gg6.rand();
	//	x(i, 6) = gg7.rand();

	//	y(i, 0) = gg8.rand();
	//	y(i, 1) = gg9.rand();
	//	y(i, 2) = gg10.rand();
	//	y(i, 3) = gg11.rand();
	//	y(i, 4) = gg12.rand();
	//	y(i, 5) = gg13.rand();
	//	y(i, 6) = gg14.rand();

	//	z(i, 0) = gg15.rand();
	//	z(i, 1) = gg16.rand();
	//	z(i, 2) = gg17.rand();
	//	z(i, 3) = gg18.rand();
	//	z(i, 4) = gg19.rand();
	//	z(i, 5) = gg20.rand();
	//	z(i, 6) = 0;
	//}
	//std::vector<std::string> head;
	//head.push_back("a");
	//head.push_back("b");
	//head.push_back("c");
	//head.push_back("d");
	//head.push_back("e");
	//head.push_back("f");
	//x.print_csv("gg1.csv", head);
	//y.print_csv("gg2.csv", head);
	//z.print_csv("gg3.csv", head);
	//exit(0);

	if (load_model != "")
	{
		try
		{
			if (!LiNGAM.load(load_model, loss_data_load))
			{
				printf("ERROR:load_model\n");
				return -1;
			}
			printf("load_model ok.\n");
			double c_factors = LiNGAM.residual_error_independ.Max();
			LiNGAM.residual_error_independ.print_csv("confounding_factors_info.csv");
			FILE* fp = fopen("confounding_factors.txt", "w");
			if (fp)
			{
				fprintf(fp, "%f\n", c_factors);
				fclose(fp);
			}

		}
		catch (std::exception& e)
		{
			printf("LiNGAM load exception:%s\n", e.what());
			return -1;
		}
	}
	else
	{
		//http://www.ar.sanken.osaka-u.ac.jp/~sshimizu/papers/JJSS08.pdf
		//	十分大きい標本サイズを集める必要がある. 
		//　Sarela and Vigario(2003) は経験上, 
		//　推定するパラメータ数の少なくとも 5 倍(T > 5m^2) は必要.

		if (confounding_factors)
		{
			std::vector<int> knowledge;
			prior_knowledge(prior_knowledge_file.c_str(), header_names, knowledge);

			LiNGAM.early_stopping = early_stopping;
			LiNGAM.distribution_rate = distribution_rate;
			LiNGAM.temperature_alp = temperature_alp;
			LiNGAM.prior_knowledge = knowledge;
			LiNGAM.prior_knowledge_rate = prior_knowledge_rate;
			LiNGAM.confounding_factors_sampling = confounding_factors_sampling;
			LiNGAM.confounding_factors_upper = confounding_factors_upper;
			LiNGAM.rho = rho;

			LiNGAM.fit2(xs, max_ica_iteration, ica_tolerance);
		}
		else
		{
			LiNGAM.fit(xs, max_ica_iteration, ica_tolerance);
		}
		
		LiNGAM.save(std::string("lingam.model"), true);
	}

	LiNGAM.B.print_e("(#1)B");
	if (lasso)
	{
		if (LiNGAM.remove_redundancy(lasso, lasso_itr_max, lasso_tol) != 0)
		{
			printf("ERROR:lasso(remove_redundancy)\n");
		}
	}
	LiNGAM.before_sorting(LiNGAM.mutual_information);
	LiNGAM.before_sorting();

	LiNGAM.B.print_e("(#2)B");
	if (min_cor_delete > 0)
	{
		Matrix<dnn_double> XCor = xs.Cor();
		for (int i = 0; i < LiNGAM.B.m; i++)
		{
			for (int j = 0; j < LiNGAM.B.n; j++)
			{
				if (fabs(XCor(i, j)) < min_cor_delete)
				{
					LiNGAM.B(i, j) = 0.0;
				}
			}
		}
	}

	if (min_delete > 0 && min_delete_srt == 0)
	{
		for (int i = 0; i < LiNGAM.B.m; i++)
		{
			for (int j = 0; j < LiNGAM.B.n; j++)
			{
				if (fabs(LiNGAM.B(i, j)) < min_delete)
				{
					LiNGAM.B(i, j) = 0.0;
				}
			}
		}
	}

	if ( min_delete_srt > 0 )
	{
		struct b_data_t
		{
			double b;
			int Cause;
			int result;

			bool operator<(const struct b_data_t& right) const {
				return fabs(b) <= fabs(right.b);
			}
		};
		std::vector< struct b_data_t> b_data;
		for (int i = 0; i < LiNGAM.B.m; i++)
		{
			for (int j = 0; j < LiNGAM.B.n; j++)
			{
				if (fabs(LiNGAM.B(i, j)) < 1.0e-6)
				{
					continue;
				}

				struct b_data_t b;
				b.b = LiNGAM.B(i, j);
				b.result = i;
				b.Cause = j;
				b_data.push_back(b);
			}
		}

		std::sort(b_data.begin(), b_data.end());
		for (int i = 0; i < b_data.size(); i++)
		{
			printf("[%d]b(%d,%d)=%f\n", i, b_data[i].result, b_data[i].Cause, b_data[i].b);
		}
		if (min_delete_srt < b_data.size())
		{
			for (int i = 0; i < min_delete_srt; i++)
			{
				int ii = b_data[i].result;
				int jj = b_data[i].Cause;
				LiNGAM.B(ii, jj) = 0.0;
			}
		}
	}

	//mutual_information_cut = 0.001;
	if (mutual_information_cut > 0)
	{
		for (int i = 0; i < LiNGAM.B.m; i++)
		{
			for (int j = 0; j < LiNGAM.B.n; j++)
			{
				if (LiNGAM.mutual_information(i,j) < mutual_information_cut)
				{
					LiNGAM.B(i, j) = 0.0;
				}
			}
		}
	}
	if (cor_range[0] != 0 && cor_range[1] != 0 && cor_range[0] < cor_range[1])
	{
		Matrix<dnn_double> XCor = xs.Cor();
		for (int i = 0; i < LiNGAM.B.m; i++)
		{
			for (int j = 0; j < LiNGAM.B.n; j++)
			{
				if (fabs(XCor(i, j)) < cor_range[0])
				{
					LiNGAM.B(i, j) = 0.0;
				}
				if (fabs(XCor(i, j)) > cor_range[1])
				{
					LiNGAM.B(i, j) = 0.0;
				}
			}
		}
	}
	LiNGAM.B.print_e("(#3)B");

	std::vector<int> residual_flag(xs.n, 0);
	if (error_distr)
	{
		LiNGAM.residual_error.print_csv("error_distr.csv", header_names);
		std::vector<std::string> shapiro_wilk_values;
		Shapiro_wilk_test(LiNGAM.residual_error, header_names, residual_flag, shapiro_wilk_values);

#ifdef USE_GNUPLOT
		gnuPlot plot1(std::string(GNUPLOT_PATH), 3);
		if (capture)
		{
			plot1.set_capture(error_distr_size, std::string("causal_multi_histgram.png"));
		}
		plot1.multi_histgram(std::string("causal_multi_histgram.png"), LiNGAM.residual_error, header_names, residual_flag);
		plot1.draw();
#endif
	}
	
	if (y_var.size())
	{
		LiNGAM.linear_regression_var.push_back(y_var[0]);
	}
	LiNGAM.diagram(header_names, y_var, residual_flag, "digraph.txt", sideways, diaglam_size, output_diaglam_type, false, mutual_information_cut, view_confounding_factors);
	LiNGAM.report(header_names);

	{
		std::vector<std::string>& var = LiNGAM.linear_regression_var;

		std::vector<int> var_index;
		if (var.size())
		{
			var_index.resize(var.size());
			for (int i = 0; i < var.size(); i++)
			{
				for (int j = 0; j < header_names_org.size(); j++)
				{
					if (var[i] == header_names_org[j])
					{
						var_index[i] = j;
					}
					else if ("\"" + var[i] + "\"" == header_names_org[j])
					{
						var_index[i] = j;
					}
				}
			}
		}

		FILE* fp = fopen("select_variables.dat", "w");
		if (fp)
		{
			for (int i = 0; i < var_index.size(); i++)
			{
				fprintf(fp, "%d,%s\n", var_index[i], var[i].c_str());
			}
			fclose(fp);
		}
	}

	for (int i = 0; i < LiNGAM.intercept.m; i++)
	{
		printf("intercept[%s]:%.4f\n", header_names[i].c_str(), LiNGAM.intercept(i, 0));
	}
	for (int i = 0; i < LiNGAM.residual_error.n; i++)
	{
		printf("epsilon_mean[%s]:%.4f\n", header_names[i].c_str(), LiNGAM.residual_error.Col(i).Mean().v[0]);
	}

	for (int i = 0; i < LiNGAM.residual_error.n; i++)
	{
		for (int j = i+1; j < LiNGAM.residual_error.n; j++)
		{
			printf("residual_error_independ[%s,%s]:%.4f\n", header_names[i].c_str(), header_names[j].c_str(), LiNGAM.residual_error_independ(i,j));
		}
	}
	if (confounding_factors)
	{
		printf("Iter:%d early_stopping:%d\n", confounding_factors_sampling, early_stopping);
		printf("loss:[%.4f]\n", LiNGAM.loss_value);
	}
	printf("residual_independence:[%.4f,%.4f]\n", LiNGAM.residual_error_independ.Min(), LiNGAM.residual_error_independ.Max());
	printf("residual_error:[%.4f, %.4f]\n", LiNGAM.residual_error.Min(), LiNGAM.residual_error.Max());

	if (resp == 0)
	{
		for (int i = 0; i < argc; i++)
		{
			delete[] argv[i];
		}
		delete argv;
	}
	if (pause != 0)
	{
		printf("... pause ...[Hit enter key]\n");
		fflush(stdout);
		getchar();
	}
	return 0;
}