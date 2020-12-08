#ifndef _BREAK_SOLVER
#define _BREAK_SOLVER

inline bool is_stopping_solver()
{
	std::ifstream ifs("_stopping_solver_");
	bool ret =  ifs.is_open();
	remove("_stopping_solver_");
	return ret;
}
inline void clear_stopping_solver()
{
	if (is_stopping_solver())
	{
		remove("_stopping_solver_");
	}
}

#endif
