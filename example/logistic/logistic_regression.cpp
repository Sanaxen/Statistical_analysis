#include <stdio.h>
#include <stdlib.h>

#include <string>
#include <vector>


#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#include "linear.h"

#define _cublas_Init_def
#include "../../include/Matrix.hpp"
#include "../../include/util/csvreader.h"

#define _USE_MATH_DEFINES
#include "../../include/util/mathutil.h"

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
#define INF HUGE_VAL

static int(*info)(const char *fmt, ...) = &printf;
void parse_command_line(int argc, char **argv, char *input_file_name, char *model_file_name);
void read_problem(const char *filename);
void do_cross_validation();
void do_find_parameters();

struct feature_node *x_space;
struct parameter param;
struct problem prob;
struct model* model_;
int flag_cross_validation;
int flag_find_parameters;
int flag_C_specified;
int flag_p_specified;
int flag_solver_specified;
int nr_fold;
double bias;

struct feature_node *x;
int max_nr_attr = 64;

int flag_predict_probability = 0;

static char *line = NULL;
static int max_line_len;

char* cross_validation_file = NULL;
double Pearson_residuals = 0.0;
std::vector<double> SEs;

int printf_null(const char *s, ...) { return 0; }
void print_null(const char *s) {}

void exit_input_error(int line_num)
{
	fprintf(stderr, "Wrong input format at line %d\n", line_num);
	exit(1);
}



static char* readline(FILE *input)
{
	int len;

	if (fgets(line, max_line_len, input) == NULL)
		return NULL;

	while (strrchr(line, '\n') == NULL)
	{
		max_line_len *= 2;
		line = (char *)realloc(line, max_line_len);
		len = (int)strlen(line);
		if (fgets(line + len, max_line_len - len, input) == NULL)
			break;
	}
	return line;
}

void train_exit_with_help()
{
	printf(
		"Usage: train [options] training_set_file [model_file]\n"
		"options:\n"
		"-s type : set type of solver (default 1)\n"
		"  for multi-class classification\n"
		"	 0 -- L2-regularized logistic regression (primal)\n"
		"	 1 -- L2-regularized L2-loss support vector classification (dual)\n"
		"	 2 -- L2-regularized L2-loss support vector classification (primal)\n"
		"	 3 -- L2-regularized L1-loss support vector classification (dual)\n"
		"	 4 -- support vector classification by Crammer and Singer\n"
		"	 5 -- L1-regularized L2-loss support vector classification\n"
		"	 6 -- L1-regularized logistic regression\n"
		"	 7 -- L2-regularized logistic regression (dual)\n"
		"  for regression\n"
		"	11 -- L2-regularized L2-loss support vector regression (primal)\n"
		"	12 -- L2-regularized L2-loss support vector regression (dual)\n"
		"	13 -- L2-regularized L1-loss support vector regression (dual)\n"
		"-c cost : set the parameter C (default 1)\n"
		"-p epsilon : set the epsilon in loss function of SVR (default 0.1)\n"
		"-e epsilon : set tolerance of termination criterion\n"
		"	-s 0 and 2\n"
		"		|f'(w)|_2 <= eps*min(pos,neg)/l*|f'(w0)|_2,\n"
		"		where f is the primal function and pos/neg are # of\n"
		"		positive/negative data (default 0.01)\n"
		"	-s 11\n"
		"		|f'(w)|_2 <= eps*|f'(w0)|_2 (default 0.0001)\n"
		"	-s 1, 3, 4, and 7\n"
		"		Dual maximal violation <= eps; similar to libsvm (default 0.1)\n"
		"	-s 5 and 6\n"
		"		|f'(w)|_1 <= eps*min(pos,neg)/l*|f'(w0)|_1,\n"
		"		where f is the primal function (default 0.01)\n"
		"	-s 12 and 13\n"
		"		|f'(alpha)|_1 <= eps |f'(alpha0)|,\n"
		"		where f is the dual function (default 0.1)\n"
		"-B bias : if bias >= 0, instance x becomes [x; bias]; if < 0, no bias term added (default -1)\n"
		"-wi weight: weights adjust the parameter C of different classes (see README for details)\n"
		"-v n: n-fold cross validation mode\n"
		"-C : find parameters (C for -s 0, 2 and C, p for -s 11)\n"
		"-q : quiet mode (no outputs)\n"
	);
	exit(1);
}
void predict_exit_with_help()
{
	printf(
		"Usage: predict [options] test_file model_file output_file\n"
		"options:\n"
		"-b probability_estimates: whether to output probability estimates, 0 or 1 (default 0); currently for logistic regression only\n"
		"-q : quiet mode (no outputs)\n"
	);
	exit(1);
}


int logistic_regression_train(int argc, char **argv)
{
	char input_file_name[1024];
	char model_file_name[1024];
	const char *error_msg;

	parse_command_line(argc, argv, input_file_name, model_file_name);
	read_problem(input_file_name);
	error_msg = check_parameter(&prob, &param);

	if (error_msg)
	{
		fprintf(stderr, "ERROR: %s\n", error_msg);
		exit(1);
	}

	if (flag_find_parameters)
	{
		do_find_parameters();
	}
	else if (flag_cross_validation)
	{
		do_cross_validation();
	}
	else
	{
		model_ = train(&prob, &param);
		if (save_model(model_file_name, model_))
		{
			fprintf(stderr, "can't save model to file %s\n", model_file_name);
			exit(1);
		}
		free_and_destroy_model(&model_);
	}
	destroy_param(&param);
	free(prob.y);
	free(prob.x);
	free(x_space);
	free(line);

	return 0;
}

void do_find_parameters()
{
	double start_C, start_p, best_C, best_p, best_score;
	if (flag_C_specified)
		start_C = param.C;
	else
		start_C = -1.0;
	if (flag_p_specified)
		start_p = param.p;
	else
		start_p = -1.0;

	printf("Doing parameter search with %d-fold cross validation.\n", nr_fold);
	find_parameters(&prob, &param, nr_fold, start_C, start_p, &best_C, &best_p, &best_score);
	if (param.solver_type == L2R_LR || param.solver_type == L2R_L2LOSS_SVC)
		printf("Best C = %g  CV accuracy = %g%%\n", best_C, 100.0*best_score);
	else if (param.solver_type == L2R_L2LOSS_SVR)
		printf("Best C = %g Best p = %g  CV MSE = %g\n", best_C, best_p, best_score);
}

void do_cross_validation()
{
	int i;
	int total_correct = 0;
	double total_error = 0;
	double sumv = 0, sumy = 0, sumvv = 0, sumyy = 0, sumvy = 0;
	double *target = Malloc(double, prob.l);
	/***********************/
	//** To caluculate precision/recall for each class **/
	int tp = 0;
	int fp = 0;
	int tn = 0;
	int fn = 0;
	FILE* file_p = stdout;
	if (cross_validation_file)
	{
		file_p = fopen(cross_validation_file, "w");
		if (file_p == NULL)
		{
			file_p = stdout;
		}
	}
	/***********************/

	cross_validation(&prob, &param, nr_fold, target);
	if (param.solver_type == L2R_L2LOSS_SVR ||
		param.solver_type == L2R_L1LOSS_SVR_DUAL ||
		param.solver_type == L2R_L2LOSS_SVR_DUAL)
	{
		for (i = 0; i<prob.l; i++)
		{
			double y = prob.y[i];
			double v = target[i];
			total_error += (v - y)*(v - y);
			sumv += v;
			sumy += y;
			sumvv += v*v;
			sumyy += y*y;
			sumvy += v*y;
		}
		printf("Cross Validation Mean squared error = %g\n", total_error / prob.l);
		printf("Cross Validation Squared correlation coefficient = %g\n",
			((prob.l*sumvy - sumv*sumy)*(prob.l*sumvy - sumv*sumy)) /
			((prob.l*sumvv - sumv*sumv)*(prob.l*sumyy - sumy*sumy))
		);
	}
	else
	{
#if 0
		for (i = 0; i<prob.l; i++)
			if (target[i] == prob.y[i])
				++total_correct;
		printf("Cross Validation Accuracy = %g%%\n", 100.0*total_correct / prob.l);
#else
		for (i = 0; i<prob.l; i++) {
			if (prob.y[i] == 1) { // True label = +1
			if (target[i] == prob.y[i]) {
					tp++;
				}
				else {
					fp++;
				}
			}
			else { // True label = -1
				if (target[i] == prob.y[i]) {
					tn++;
				}
				else {
					fn++;
				}
			}
		}
		printf("Cross Validation Accuracy = %.3f%%\n", 100.0 * ((double)(tp + tn) / (double)(tp + fp + tn + fn)));
		
		// Precision and recall
		double pos_prec = ((double)tp / (double)(tp + fp));
		double pos_rec = ((double)tp / (double)(tp + fn));
		double pos_f1 = (2 * pos_prec * pos_rec) / (pos_prec + pos_rec);

		double neg_prec = ((double)tn / (double)(tn + fn));
		double neg_rec = ((double)tn / (double)(tn + fp));
		double neg_f1 = (2 * neg_prec * neg_rec) / (neg_prec + neg_rec);

		fprintf( file_p, "Positive (+1) class:\n");
		fprintf(file_p, "  precision = %.3f\n", pos_prec);
		fprintf(file_p, "     recall = %.3f\n", pos_rec);
		fprintf(file_p, "   F1 value = %.3f\n\n", pos_f1);
		fprintf(file_p, "Negative (-1) class:\n");
		fprintf(file_p, "  precision = %.3f\n", neg_prec);
		fprintf(file_p, "     recall = %.3f\n", neg_rec);
		fprintf(file_p, "   F1 value = %.3f\n\n", neg_f1);
		if (file_p != stdout)
		{
			fclose(file_p);
		}
#endif

	}

	free(target);
}

void parse_command_line(int argc, char **argv, char *input_file_name, char *model_file_name)
{
	int i;
	void(*print_func)(const char*) = NULL;	// default printing to stdout

											// default values
	param.solver_type = L2R_L2LOSS_SVC_DUAL;
	param.C = 1;
	param.eps = INF; // see setting below
	param.p = 0.1;
	param.nr_weight = 0;
	param.weight_label = NULL;
	param.weight = NULL;
	param.init_sol = NULL;
	flag_cross_validation = 0;
	flag_C_specified = 0;
	flag_p_specified = 0;
	flag_solver_specified = 0;
	flag_find_parameters = 0;
	bias = -1;

	// parse options
	for (i = 1; i<argc; i++)
	{
		if (argv[i][0] != '-') break;
		if (++i >= argc)
			train_exit_with_help();
		switch (argv[i - 1][1])
		{
		case 's':
			param.solver_type = atoi(argv[i]);
			flag_solver_specified = 1;
			break;

		case 'c':
			param.C = atof(argv[i]);
			flag_C_specified = 1;
			break;

		case 'p':
			flag_p_specified = 1;
			param.p = atof(argv[i]);
			break;

		case 'e':
			param.eps = atof(argv[i]);
			break;

		case 'B':
			bias = atof(argv[i]);
			break;

		case 'w':
			++param.nr_weight;
			param.weight_label = (int *)realloc(param.weight_label, sizeof(int)*param.nr_weight);
			param.weight = (double *)realloc(param.weight, sizeof(double)*param.nr_weight);
			param.weight_label[param.nr_weight - 1] = atoi(&argv[i - 1][2]);
			param.weight[param.nr_weight - 1] = atof(argv[i]);
			break;

		case 'v':
			flag_cross_validation = 1;
			nr_fold = atoi(argv[i]);
			if (nr_fold < 2)
			{
				fprintf(stderr, "n-fold cross validation: n must >= 2\n");
				train_exit_with_help();
			}
			break;

		case 'q':
			print_func = &print_null;
			i--;
			break;

		case 'C':
			flag_find_parameters = 1;
			i--;
			break;

		default:
			fprintf(stderr, "unknown option: -%c\n", argv[i - 1][1]);
			train_exit_with_help();
			break;
		}
	}

	set_print_string_function(print_null);

	// determine filenames
	if (i >= argc)
		train_exit_with_help();

	strcpy(input_file_name, argv[i]);

	if (i<argc - 1)
		strcpy(model_file_name, argv[i + 1]);
	else
	{
		char *p = strrchr(argv[i], '/');
		if (p == NULL)
			p = argv[i];
		else
			++p;
		sprintf(model_file_name, "%s.model", p);
	}

	// default solver for parameter selection is L2R_L2LOSS_SVC
	if (flag_find_parameters)
	{
		if (!flag_cross_validation)
			nr_fold = 5;
		if (!flag_solver_specified)
		{
			fprintf(stderr, "Solver not specified. Using -s 2\n");
			param.solver_type = L2R_L2LOSS_SVC;
		}
		else if (param.solver_type != L2R_LR && param.solver_type != L2R_L2LOSS_SVC && param.solver_type != L2R_L2LOSS_SVR)
		{
			fprintf(stderr, "Warm-start parameter search only available for -s 0, -s 2 and -s 11\n");
			train_exit_with_help();
		}
	}

	if (param.eps == INF)
	{
		switch (param.solver_type)
		{
		case L2R_LR:
		case L2R_L2LOSS_SVC:
			param.eps = 0.01;
			break;
		case L2R_L2LOSS_SVR:
			param.eps = 0.0001;
			break;
		case L2R_L2LOSS_SVC_DUAL:
		case L2R_L1LOSS_SVC_DUAL:
		case MCSVM_CS:
		case L2R_LR_DUAL:
			param.eps = 0.1;
			break;
		case L1R_L2LOSS_SVC:
		case L1R_LR:
			param.eps = 0.01;
			break;
		case L2R_L1LOSS_SVR_DUAL:
		case L2R_L2LOSS_SVR_DUAL:
			param.eps = 0.1;
			break;
		}
	}
}

// read in a problem (in libsvm format)
void read_problem(const char *filename)
{
	int max_index, inst_max_index, i;
	size_t elements, j;
	FILE *fp = fopen(filename, "r");
	char *endptr;
	char *idx, *val, *label;

	if (fp == NULL)
	{
		fprintf(stderr, "can't open input file %s\n", filename);
		exit(1);
	}

	prob.l = 0;
	elements = 0;
	max_line_len = 1024;
	line = Malloc(char, max_line_len);
	while (readline(fp) != NULL)
	{
		char *p = strtok(line, " \t"); // label

									   // features
		while (1)
		{
			p = strtok(NULL, " \t");
			if (p == NULL || *p == '\n') // check '\n' as ' ' may be after the last feature
				break;
			elements++;
		}
		elements++; // for bias term
		prob.l++;
	}
	rewind(fp);

	prob.bias = bias;

	prob.y = Malloc(double, prob.l);
	prob.x = Malloc(struct feature_node *, prob.l);
	x_space = Malloc(struct feature_node, elements + prob.l);

	max_index = 0;
	j = 0;
	for (i = 0; i<prob.l; i++)
	{
		inst_max_index = 0; // strtol gives 0 if wrong format
		readline(fp);
		prob.x[i] = &x_space[j];
		label = strtok(line, " \t\n");
		if (label == NULL) // empty line
			exit_input_error(i + 1);

		prob.y[i] = strtod(label, &endptr);
		if (endptr == label || *endptr != '\0')
			exit_input_error(i + 1);

		while (1)
		{
			idx = strtok(NULL, ":");
			val = strtok(NULL, " \t");

			if (val == NULL)
				break;

			errno = 0;
			x_space[j].index = (int)strtol(idx, &endptr, 10);
			if (endptr == idx || errno != 0 || *endptr != '\0' || x_space[j].index <= inst_max_index)
				exit_input_error(i + 1);
			else
				inst_max_index = x_space[j].index;

			errno = 0;
			x_space[j].value = strtod(val, &endptr);
			if (endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
				exit_input_error(i + 1);

			++j;
		}

		if (inst_max_index > max_index)
			max_index = inst_max_index;

		if (prob.bias >= 0)
			x_space[j++].value = prob.bias;

		x_space[j++].index = -1;
	}

	if (prob.bias >= 0)
	{
		prob.n = max_index + 1;
		for (i = 1; i<prob.l; i++)
			(prob.x[i] - 2)->index = prob.n;
		x_space[j - 2].index = prob.n;
	}
	else
		prob.n = max_index;

	fclose(fp);
}



void do_predict(FILE *input, FILE *output)
{
	int correct = 0;
	int total = 0;
	double error = 0;
	double sump = 0, sumt = 0, sumpp = 0, sumtt = 0, sumpt = 0;

	int nr_class = get_nr_class(model_);
	double *prob_estimates = NULL;
	int j, n;
	int nr_feature = get_nr_feature(model_);
	if (model_->bias >= 0)
		n = nr_feature + 1;
	else
		n = nr_feature;

	if (flag_predict_probability)
	{
		int *labels;

		if (!check_probability_model(model_))
		{
			fprintf(stderr, "probability output is only supported for logistic regression\n");
			exit(1);
		}

		labels = (int *)malloc(nr_class * sizeof(int));
		get_labels(model_, labels);
		prob_estimates = (double *)malloc(nr_class * sizeof(double));
		fprintf(output, "labels");
		for (j = 0; j<nr_class; j++)
			fprintf(output, " %d", labels[j]);
		fprintf(output, "\n");
		free(labels);
	}

	std::vector<double>Pearson;
	std::vector<double>target_label_list;
	double mean = 0.0;

	max_line_len = 1024;
	line = (char *)malloc(max_line_len * sizeof(char));
	while (readline(input) != NULL)
	{
		int i = 0;
		double target_label, predict_label;
		char *idx, *val, *label, *endptr;
		int inst_max_index = 0; // strtol gives 0 if wrong format

		label = strtok(line, " \t\n");
		if (label == NULL) // empty line
			exit_input_error(total + 1);

		target_label = strtod(label, &endptr);
		if (endptr == label || *endptr != '\0')
			exit_input_error(total + 1);

		while (1)
		{
			if (i >= max_nr_attr - 2)	// need one more for index = -1
			{
				max_nr_attr *= 2;
				x = (struct feature_node *) realloc(x, max_nr_attr * sizeof(struct feature_node));
			}

			idx = strtok(NULL, ":");
			val = strtok(NULL, " \t");

			if (val == NULL)
				break;
			errno = 0;
			x[i].index = (int)strtol(idx, &endptr, 10);
			if (endptr == idx || errno != 0 || *endptr != '\0' || x[i].index <= inst_max_index)
				exit_input_error(total + 1);
			else
				inst_max_index = x[i].index;

			errno = 0;
			x[i].value = strtod(val, &endptr);
			if (endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
				exit_input_error(total + 1);

			// feature indices larger than those in training are not used
			if (x[i].index <= nr_feature)
				++i;
		}

		if (model_->bias >= 0)
		{
			x[i].index = n;
			x[i].value = model_->bias;
			i++;
		}
		x[i].index = -1;

		if (flag_predict_probability)
		{
			int j;
			predict_label = predict_probability(model_, x, prob_estimates);

			if (target_label == predict_label)
			{
				fprintf(output, "o ");
			}
			else
			{
				fprintf(output, "x ");
			}
			fprintf(output, "%.3f ", target_label);
			fprintf(output, "%.3f", predict_label);
			for (j = 0; j<model_->nr_class; j++)
				fprintf(output, " %.3f", prob_estimates[j]);
			fprintf(output, "\n");

			double d = target_label - prob_estimates[0];
			d = d*d / ((prob_estimates[0]+1.0e-10) * (1.0 - (prob_estimates[0] + 1.0e-10)));

			Pearson.push_back(d);
		}
		else
		{
			predict_label = predict(model_, x);
			fprintf(output, "%.17g\n", predict_label);
		}

		if (predict_label == target_label)
			++correct;
		error += (predict_label - target_label)*(predict_label - target_label);
		sump += predict_label;
		sumt += target_label;
		sumpp += predict_label*predict_label;
		sumtt += target_label*target_label;
		sumpt += predict_label*target_label;

		target_label_list.push_back(target_label);
		mean += target_label;
		++total;
	}
	mean /= total;

	double sigma = 0.0;
	for (int i = 0; i < target_label_list.size(); i++)
	{
		sigma += (target_label_list[i] - mean)*(target_label_list[i] - mean);
	}
	sigma /= total;

	if (check_regression_model(model_))
	{
		info("Mean squared error = %g (regression)\n", error / total);
		info("Squared correlation coefficient = %g (regression)\n",
			((total*sumpt - sump*sumt)*(total*sumpt - sump*sumt)) /
			((total*sumpp - sump*sump)*(total*sumtt - sumt*sumt))
		);
	}
	else
	{
		Pearson_residuals = 0.0;
		for (int i = 0; i < Pearson.size(); i++)
		{
			Pearson_residuals += Pearson[i];
		}
		double se = error;
		info("Accuracy = %.3f%% (%d/%d)\n", (double)correct / total * 100, correct, total);
		info("SE = %.3f (regression)\n", se);
		info("MSE = %.3f (regression)\n", error / total);
		info("Squared correlation coefficient = %.3f (regression)\n",
			((total*sumpt - sump*sumt)*(total*sumpt - sump*sumt)) /
			((total*sumpp - sump*sump)*(total*sumtt - sumt*sumt))
		);
		info("Pearson_residuals = %.3f (regression)\n", Pearson_residuals);

		double α = 0.05;
		info("Confidence interval=%.3f\n", α);

		Standard_normal_distribution snd;
		double za = 1.96;
		if (α == 0.1) za = 1.645;
		else if (α == 0.05) za = 1.96;
		else if (α == 0.01) za = 2.576;
		else za = snd.p_value(α / 2);
		info("Za:%.3f\n", za);


		//https://istat.co.jp/ta_commentary/logistic_04
		//Wald–square
		Chi_distribution chi_distribution(1);
		double x = chi_distribution.p_value(α);
		info("z^2>%.3f\n", x);

		//printf("0.320354==%f\n", 1.0-chi_distribution.distribution(0.9875, &x));
		if (model_->bias < 0)
		{
			for (int i = 0; i < model_->nr_feature; i++)
			{
				double tmp;
				int ii = i + 1;
				double Wald_square = (model_->w[i] / SEs[ii])*(model_->w[i] / SEs[ii]);
				double chi_pdf = 1.0 - chi_distribution.distribution(Wald_square, &tmp);
				info("w=%.3f SE=%.3f Wald=%.3g p-value=%.4f"
					" down=%.3f up=%.3f\n", 
					model_->w[i], SEs[ii], Wald_square, chi_pdf,
					model_->w[i]-za* SEs[ii], model_->w[i] + za * SEs[ii]);
			}
		}
		else
		{
			for (int i = 0; i <= model_->nr_feature; i++)
			{
				double tmp;
				int ii = i + 1;
				if (i == model_->nr_feature) ii = 0;
				double Wald_square = (model_->w[i] / SEs[ii])*(model_->w[i] / SEs[ii]);
				double chi_pdf = 1.0 - chi_distribution.distribution(Wald_square, &tmp);
				info("w=%.3f SE=%.3f Wald=%.3g p-value=%.4f"
					" down=%.3f up=%.3f\n",
					model_->w[i], SEs[ii], Wald_square, chi_pdf,
					model_->w[i] - za * SEs[ii], model_->w[i] + za * SEs[ii]);
			}
		}
	}
	if (flag_predict_probability)
		free(prob_estimates);
}


int logistic_regression_predict(int argc, char **argv)
{
	FILE *input, *output;
	int i;

	// parse options
	for (i = 1; i<argc; i++)
	{
		if (argv[i][0] != '-') break;
		++i;
		switch (argv[i - 1][1])
		{
		case 'b':
			flag_predict_probability = atoi(argv[i]);
			break;
		case 'q':
			info = &printf_null;
			i--;
			break;
		default:
			fprintf(stderr, "unknown option: -%c\n", argv[i - 1][1]);
			predict_exit_with_help();
			break;
		}
	}
	if (i >= argc)
		predict_exit_with_help();

	input = fopen(argv[i], "r");
	if (input == NULL)
	{
		fprintf(stderr, "can't open input file %s\n", argv[i]);
		exit(1);
	}

	output = fopen(argv[i + 2], "w");
	if (output == NULL)
	{
		fprintf(stderr, "can't open output file %s\n", argv[i + 2]);
		exit(1);
	}

	if ((model_ = load_model(argv[i + 1])) == 0)
	{
		fprintf(stderr, "can't open model file %s\n", argv[i + 1]);
		exit(1);
	}



	{
		char fname[260];
		strcpy(fname, argv[i]);
		char* p = strstr(fname, "_libsvm");
		if (p)
		{
			*p = '\0';
		}
		else
		{
			printf("model file name rule:datafile.csv -> datafile.csv_libsvm****\n");
		}
		strcat(fname, ".no_header");
		printf("%s\n", fname);
		CSVReader csv1(fname, ',', false);
		Matrix<dnn_double> T = csv1.toMat();

		std::vector<double> means;
		std::vector<double> sigmas;
		for (int j = 0; j < T.n; j++)
		{
			double mean = 0.0;
			for (int i = 0; i < T.m; i++)
			{
				mean += T(i, j);
			}
			mean /= T.m;
			means.push_back(mean);
			//printf("mean[%d]:%f\n", j, mean);
		}
		for (int j = 0; j < T.n; j++)
		{
			double s = 0.0;
			for (int i = 0; i < T.m; i++)
			{
				s += (T(i, j)-means[j])*(T(i, j) - means[j]);
			}
			s /= (T.m - 1);
			sigmas.push_back(sqrt(s));
			//printf("sigmas[%d]:%f\n", j, sqrt(s));
		}
		for (int i = 0; i < T.n; i++)
		{
			double se = sqrt(sigmas[i]) / sqrt(T.m);
			SEs.push_back(se);
		}
	}
	x = (struct feature_node *) malloc(max_nr_attr * sizeof(struct feature_node));
	do_predict(input, output);

	free_and_destroy_model(&model_);
	free(line);
	free(x);
	fclose(input);
	fclose(output);
	return 0;
}


int main(int argc, char** argv)
{
	int stat = -1;
	printf("logistic_regression START\n");

	int argc2 = 0;
	char** argv2 = new char*[32];
	for (int i = 0; i < 32; i++)
	{
		argv2[i] = new char[640];
	}


	bool train = false;
	for (int count = 1; count + 1 < argc; count += 2) {
		std::string argname(argv[count]);
		if (argname == "--train") {
			train = atoi(argv[count + 1]) != 0 ? true : false;
			continue;
		}
		if (argname == "--predict") {
			train = atoi(argv[count + 1]) != 0 ? false : true;
			continue;
		}
	}

	std::string file, model, output;
	int s = 0;
	double c = 0.0;
	double bias = -1.0;
	int cross_validation = 0;

	if (train)
	{
		for (int count = 1; count + 1 < argc; count += 2) {
			std::string argname(argv[count]);
			if (argname == "--cv") {
				cross_validation = atoi(argv[count + 1]);
				continue;
			}
			if (argname == "--bias") {
				bias = atof(argv[count + 1]);
				continue;
			}
			if (argname == "--L1") {
				s = 6;
				c = atof(argv[count + 1]);
				continue;
			}
			if (argname == "--L2") {
				s = 0;
				c = atof(argv[count + 1]);
				continue;
			}
			if (argname == "--L2dual") {
				s = 7;
				c = atof(argv[count + 1]);
				continue;
			}
			if (argname == "--file") {
				file = argv[count + 1];
				continue;
			}
			if (argname == "--model") {
				model = argv[count + 1];
				continue;
			}
			if (argname == "--output") {
				output = argv[count + 1];
				continue;
			}
			if (argname == "--cv_report") {
				cross_validation_file = argv[count + 1];
				continue;
			}
		}


		strcpy(argv2[0], "");
		strcpy(argv2[1], "-s");
		sprintf(argv2[2], "%d", s);
		strcpy(argv2[3], "-B");
		sprintf(argv2[4], "%f", bias);

		if (c < 0.0001) c = 0.0001;
		strcpy(argv2[5], "-c");
		sprintf(argv2[6], "%f", c);
		argc2 = 7;

		if (cross_validation >= 2)
		{
			strcpy(argv2[argc2], "-v");
			sprintf(argv2[argc2+1], "%d", cross_validation);
			argc2 += 2;
		}
		strcpy(argv2[argc2], file.c_str()); argc2++;
		strcpy(argv2[argc2], model.c_str()); argc2++;

		{
			FILE* fp = fopen("debug_commandline1.txt", "w");
			for (int i = 0; i < argc; i++)
			{
				fprintf(fp, "%s ", argv[i]);
			}
			fclose(fp);
		}
		stat = logistic_regression_train(argc2, argv2);
	}
	else
	{
		for (int count = 1; count + 1 < argc; count += 2) {
			std::string argname(argv[count]);
			if (argname == "--file") {
				file = argv[count + 1];
				continue;
			}
			if (argname == "--model") {
				model = argv[count + 1];
				continue;
			}
			if (argname == "--output") {
				output = argv[count + 1];
				continue;
			}
		}


		strcpy(argv2[0], "");
		strcpy(argv2[1], "-b");
		sprintf(argv2[2], "%d", 1);
		strcpy(argv2[3], file.c_str());
		strcpy(argv2[4], model.c_str());
		strcpy(argv2[5], output.c_str());
		argc2 = 6;

		{
			FILE* fp = fopen("debug_commandline2.txt", "w");
			for (int i = 0; i < argc; i++)
			{
				fprintf(fp, "%s ", argv[i]);
			}
			fclose(fp);
		}
		stat = logistic_regression_predict(argc2, (char**)argv2);
	}

	for (int i = 0; i < 32; i++)
	{
		delete[] argv2[i];
	}
	delete[] argv2;

	printf("logistic_regression END\n\n");
	return stat;
}