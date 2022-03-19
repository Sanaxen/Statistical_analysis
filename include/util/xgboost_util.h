#ifndef _XGBOOST_UTIL_H

#define _XGBOOST_UTIL_H

#include "../../../third_party/xgboost/include/xgboost/c_api.h"

#pragma comment(lib, "../../third_party/xgboost/lib/dmlc.lib")
#pragma comment(lib, "../../third_party/xgboost/lib/xgboost.lib")


#define safe_xgboost(call) {   \
if ( xgboost_util_error_ == 0){  \
    xgboost_util_error_= (call); \
    if (xgboost_util_error_ != 0) { \
      fprintf(stderr, "%s:%d: error in %s: %s\n", __FILE__, __LINE__, #call, XGBGetLastError()); \
    } \
}\
}

template<class T> class xgboost_util
{
	BoosterHandle h_booster;
    DMatrixHandle h_train[1];
    Matrix<float> x;
    Matrix<float> y;
    Matrix<float> pred_y;

    int xgboost_util_error_ = 0;
public:
	xgboost_util() {
    }
    
    ~xgboost_util() {
        safe_xgboost(XGDMatrixFree(h_train[0]));
        safe_xgboost(XGBoosterFree(h_booster));
    }


	void set_train(Matrix<T>& train, Matrix<T>& label)
	{
        x = Matrix<float>(train.m, train.n);
        y = Matrix<float>(label.m, label.n);
#pragma omp parallel for
        for (int j = 0; j < train.m; j++)
        {
            for (int i = 0; i < train.n; i++)
            {
                x(i, j) = train(i, j);
            }
        }
#pragma omp parallel for
        for (int j = 0; j < label.m; j++)
        {
            for (int i = 0; i < label.n; i++)
            {
                y(i, j) = label(i, j);
            }
        }

        safe_xgboost(XGDMatrixCreateFromMat((float*)x.v, x.m, x.n, -1, &h_train[0]));
        safe_xgboost(XGDMatrixSetFloatInfo(h_train[0], "label", y.v, y.m));

        bst_ulong bst_result;
        const float* out_floats;
        safe_xgboost(XGDMatrixGetFloatInfo(h_train[0], "label", &bst_result, &out_floats));
        //for (unsigned int i = 0; i < bst_result; i++)
        //    std::cout << "label[" << i << "]=" << out_floats[i] << std::endl;

        safe_xgboost(XGBoosterCreate(h_train, 1, &h_booster));
        safe_xgboost(XGBoosterSetParam(h_booster, "seed", "0"));
        safe_xgboost(XGBoosterSetParam(h_booster, "booster", "gbtree"));
        safe_xgboost(XGBoosterSetParam(h_booster, "objective", "reg:squarederror"));
        safe_xgboost(XGBoosterSetParam(h_booster, "max_depth", "6"));
        safe_xgboost(XGBoosterSetParam(h_booster, "eta", "0.1"));
        safe_xgboost(XGBoosterSetParam(h_booster, "min_child_weight", "1"));
        safe_xgboost(XGBoosterSetParam(h_booster, "subsample", "1.0"));
        safe_xgboost(XGBoosterSetParam(h_booster, "colsample_bytree", "1"));
        safe_xgboost(XGBoosterSetParam(h_booster, "num_parallel_tree", "1"));
    }

    int train(int iterations)
    {
        for (int iter = 0; iter < iterations; iter++)
            safe_xgboost(XGBoosterUpdateOneIter(h_booster, iter, h_train[0]));

        bst_ulong num_feature = 0;
        safe_xgboost(XGBoosterGetNumFeature(h_booster, &num_feature));
        printf("num_feature: %lu\n", (unsigned long)(num_feature));

        //bst_ulong out_size = 0;
        //char const** out_features = NULL;
        //safe_xgboost(XGDMatrixGetStrFeatureInfo(h_booster, u8"feature_name", &out_size, &out_features));
        //for (bst_ulong i = 0; i < out_size; ++i) {
        //    printf("feature %lu: %s\n", i, out_features[i]);
        //}
        return xgboost_util_error_;
    }


    std::vector<T> predict(Matrix<T>& test)
    {
        pred_y = Matrix<float>(test.m, test.n);
#pragma omp parallel for
        for (int j = 0; j < test.m; j++)
        {
            for (int i = 0; i < test.n; i++)
            {
                pred_y(i, j) = test(i, j);
            }
        }

        std::vector<T> pred;
        pred.resize(test.m, 999999.0);

        DMatrixHandle h_test = NULL;

        try
        {
            safe_xgboost(XGDMatrixCreateFromMat((float*)pred_y.v, pred_y.m, pred_y.n, -1, &h_test));

            bst_ulong out_len = 0;
            const float* f = NULL;
            safe_xgboost(XGBoosterPredict(h_booster, h_test, 0, 0, false, &out_len, &f));


            if (xgboost_util_error_ == 0)
            {
#pragma omp parallel for
                for (int i = 0; i < out_len; i++)
                {
                    pred[i] = f[i];
                }
            }
            safe_xgboost(XGDMatrixFree(h_test));
        }
        catch (...)
        {
            printf("---- xgboost predict error.----\n");
        }


        return pred;
    }

};
#endif
