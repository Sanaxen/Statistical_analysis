/*
Copyright (c) 2018, Sanaxn
All rights reserved.
Use of this source code is governed by a BSD-style license that can be found
in the LICENSE file.
*/
#ifndef _DNN_UTIL_HPP

#define _DNN_UTIL_HPP

#define NOMINMAX

#pragma warning( disable : 4244)
#pragma warning( disable : 4267)
#pragma warning( push ) 
#pragma warning( disable : 4477)
#pragma warning( disable : 4819)
#include <algorithm>
#include <random>
#include <string>
#include "tiny_dnn/tiny_dnn.h"
#include "tiny_dnn/util/util.h"


#pragma warning( pop ) 
//#include "util/file_util.hpp"
//#include "image/Image.hpp"


using fc = tiny_dnn::layers::fc;
using conv = tiny_dnn::layers::conv;
using ave_pool = tiny_dnn::layers::ave_pool;
using max_pool = tiny_dnn::max_pooling_layer;
using deconv = tiny_dnn::deconvolutional_layer;
using padding = tiny_dnn::padding;
using recurrent = tiny_dnn::recurrent_layer;
using recurrent_params = tiny_dnn::recurrent_layer_parameters;

using relu = tiny_dnn::relu_layer;
using leaky_relu = tiny_dnn::leaky_relu_layer;

using softmax = tiny_dnn::softmax_layer;

using tiny_dnn::core::connection_table;

template <typename N>
inline void set_train(N &nn, const int seq_len=0, const int bptt_max = 0, tiny_dnn::core::backend_t& defaule_backend = tiny_dnn::core::backend_t::internal) {
	nn.set_netphase(tiny_dnn::net_phase::train);
	for (unsigned int i = 0; i < nn.layer_size(); i++) {
		try {
			nn.template at<tiny_dnn::dropout_layer>(i).set_context(
				tiny_dnn::net_phase::train);
		}
		catch (tiny_dnn::nn_error &err) {
		}
		try {
			nn.template at<tiny_dnn::recurrent_layer>(i).seq_len(seq_len);
			nn.template at<tiny_dnn::recurrent_layer>(i).bptt_max(bptt_max);
			nn.template at<tiny_dnn::recurrent_layer>(i).clear_state();
		}
		catch (tiny_dnn::nn_error &err) {
		}
	}

#ifdef CNN_USE_INTEL_MKL
	for (auto n : nn)
	{
		if (n->layer_type() == "fully-connected")
		{
			n->set_backend_type(defaule_backend);
		}
	}
#endif
#ifdef CNN_USE_AVX
	for (auto n : nn)
	{
		if (n->layer_type() == "recurrent-layer")
		{
			n->set_backend_type(defaule_backend);
		}
		if (n->layer_type() == "lstm-cell")
		{
			n->set_backend_type(defaule_backend);
		}
	}
#endif

}

template <typename N>
inline void set_test(N &nn, const int seq_len=0) {
	nn.set_netphase(tiny_dnn::net_phase::test);
	for (unsigned int i = 0; i < nn.layer_size(); i++) {
		try {
			nn.template at<tiny_dnn::dropout_layer>(i).set_context(
				tiny_dnn::net_phase::test);
		}
		catch (tiny_dnn::nn_error &err) {
		}
		try {
			nn.template at<tiny_dnn::recurrent_layer>(i).seq_len(seq_len);
			nn.template at<tiny_dnn::recurrent_layer>(i).bptt_max(1e9);
			nn.template at<tiny_dnn::recurrent_layer>(i).clear_state();
		}
		catch (tiny_dnn::nn_error &err) {
		}
	}
#ifdef CNN_USE_INTEL_MKL
	for (auto n : nn)
	{
		if (n->layer_type() == "fully-connected")
		{
			n->set_backend_type(tiny_dnn::core::backend_t::intel_mkl);
		}
		if (n->layer_type() == "recurrent-layer")
		{
			n->set_backend_type(tiny_dnn::core::backend_t::avx);
		}
		if (n->layer_type() == "lstm-cell")
		{
			n->set_backend_type(tiny_dnn::core::backend_t::avx);
		}
	}
#endif

#ifdef CNN_USE_AVX
	for (auto n : nn)
	{
		if (n->layer_type() == "recurrent-layer")
		{
			n->set_backend_type(tiny_dnn::core::backend_t::avx);
		}
	}
#endif
}

template <typename N>
inline void rnn_state_reset(N &nn) {
	for (unsigned int i = 0; i < nn.layer_size(); i++) {
		try {
			nn.template at<tiny_dnn::recurrent_layer>(i).clear_state();
		}
		catch (tiny_dnn::nn_error &err) {
		}
	}
}

inline size_t deconv_out_length(size_t in_length,
	size_t window_size,
	size_t stride) {
	return (size_t)ceil((float_t)(in_length)*stride + window_size - 1);
}

inline size_t deconv_out_unpadded_length(size_t in_length,
	size_t window_size,
	size_t stride,
	padding pad_type) {
	return pad_type == padding::same
		? (size_t)ceil((float_t)in_length * stride)
		: (size_t)ceil((float_t)(in_length)*stride + window_size - 1);
}

namespace tiny_dnn {
	template <typename NetType>
	class network2 : public network<NetType>
	{
		size_t input_size;
	public:

		inline void set_input_size(size_t size)
		{
			input_size = size;
		}
		inline size_t& get_input_size()
		{
			return input_size;
		}
		inline void set_netphase(net_phase phase)
		{
			network<NetType>::set_netphase(phase);
		}
		inline result test(const std::vector<vec_t> &in, const std::vector<label_t> &t) {
			return network<NetType>::test(in, t);
		}
		inline std::vector<vec_t> test(const std::vector<vec_t> &in) {
			return network<NetType>::test(in, t);
		}
		void load(const std::string &filename,
			content_type what = content_type::weights_and_model,
			file_format format = file_format::binary) {

			return network<NetType>::load(filename, what, format);
		}
		inline void save(const std::string &filename,
			content_type what = content_type::weights_and_model,
			file_format format = file_format::binary) const {
			return network<NetType>::save(filename, what, format);
		}

		/*
		Saving consumption memory.
		We are saving memory by loading necessary data
		as much as possible so as not to read a large
		amount of learning data into memory all.
		*/
		template <typename Error,
			typename Optimizer,
			typename OnBatchEnumerate,
			typename OnEpochEnumerate,
			typename LoadTensorData>
			bool fit2(Optimizer &optimizer,
				const std::vector<tensor_t> &inputs,
				const std::vector<tensor_t> &desired_outputs,
				size_t batch_size,
				int epoch,
				OnBatchEnumerate on_batch_enumerate,
				OnEpochEnumerate on_epoch_enumerate,
				LoadTensorData load_tensor_data,
				const bool reset_weights = false,
				const int n_threads = CNN_TASK_SIZE,
				const std::vector<tensor_t> &t_cost = std::vector<tensor_t>()) {
			// check_training_data(in, t);
			check_target_cost_matrix(desired_outputs, t_cost);
			set_netphase(net_phase::train);
			net_.setup(reset_weights);

			for (auto n : net_) n->set_parallelize(true);
			optimizer.reset();
			stop_training_ = false;
			in_batch_.resize(batch_size);
			t_batch_.resize(batch_size);
			for (int iter = 0; iter < epoch && !stop_training_; iter++) {
				int k = 0;
				for (size_t i = 0; i < input_size && !stop_training_;
					i += batch_size, k += batch_size) {

					if (k + batch_size > inputs.size())
					{
						load_tensor_data();
						k = 0;
					}
					train_once<Error>(
						optimizer, &inputs[k], &desired_outputs[k],
						static_cast<int>(std::min(batch_size, (size_t)inputs.size() - k)),
						n_threads, get_target_cost_sample_pointer(t_cost, k));
					on_batch_enumerate();

					/* if (i % 100 == 0 && layers_.is_exploded()) {
					std::cout << "[Warning]Detected infinite value in weight. stop
					learning." << std::endl;
					return false;
					} */
				}
				on_epoch_enumerate();
			}
			set_netphase(net_phase::test);
			return true;
		}

		inline void normalize_tensor(const std::vector<tensor_t> &inputs,
			std::vector<tensor_t> &normalized) {
			network<NetType>::normalize_tensor(inputs, normalized);
		}

		inline void normalize_tensor(const std::vector<vec_t> &inputs,
			std::vector<tensor_t> &normalized) {
			network<NetType>::normalize_tensor(inputs, normalized);
		}

		inline void normalize_tensor(const std::vector<label_t> &inputs,
			std::vector<tensor_t> &normalized) {
			network<NetType>::normalize_tensor(inputs, normalized);
		}

	};
}

class DNNParameter
{
public:
	size_t seq_len = 10;
	float_t learning_rate = 1;
	size_t n_train_epochs = 30;
	std::string  data_dir_path = "";
	size_t n_minibatch = 16;
	tiny_dnn::core::backend_t backend_type = tiny_dnn::core::default_engine();

	int plot = 0;
	float_t on_memory_rate = 0.8f;
	float_t augmentation_rate = 0.2f;
	size_t test_sample = 30;
	size_t decay_iter = 100;
	size_t save_iter = 10;
	bool out_of_core = false;
};

class LayerInfo
{
	using padding = tiny_dnn::padding;

	size_t out_w;
	size_t out_h;
	size_t out_map;
	tiny_dnn::core::backend_t      backend_type;
	size_t parame_num = 0;
public:
	inline size_t get_parameter_num() const
	{
		return parame_num;
	}
#define PARAMETER_NUM(l) do{\
		size_t param_ = 0;\
		std::vector<tiny_dnn::vec_t *> weights = l.weights(); \
		for (int i = 0; i < weights.size(); i++)\
		{\
			tiny_dnn::vec_t &w = *weights[i]; \
			parame_num += w.size(); \
			param_ += w.size(); \
		}\
		printf("param_:%d\n", param_);\
	}while(0)

	inline void _editInfo(size_t out_w_, size_t out_h_, size_t out_map_)
	{
		out_w = out_w_;
		out_h = out_h_;
		out_map = out_map_;
	}
	inline LayerInfo(size_t iw, size_t ih, size_t imap, tiny_dnn::core::backend_t backend_type_ = tiny_dnn::core::default_engine())
	{
		out_w = iw;
		out_h = ih;
		out_map = imap;
		backend_type = backend_type_;
		printf("input %zdx%zd(=%zd) fmap:%zd\n", iw, ih, iw*ih, imap);
	}

	void set_backend_type(tiny_dnn::core::backend_t  backend_type_ = tiny_dnn::core::default_engine())
	{
		backend_type = backend_type_;
	}

	inline size_t out(size_t& s) const
	{
		return out_w*out_h*out_map;
	}
#define ACTIVATIN_SYMBL(name) \
	{\
		if (name == "selu_layer")	return selu_layer();\
		if (name == "relu")			return relu();\
		if (name == "leaky_relu")	return leaky_relu();\
		if (name == "elu")			return elu();\
		if (name == "tanh")			return tanh();\
		if (name == "sigmoid")		return sigmoid();\
		if (name == "softmax")		return softmax();\
		if (name == "softplus")		return softplus();\
		if (name == "softsign")		return softsign();\
	}

#define ACTIVATIN_FUNC(class_name) \
	{\
		size_t in_w = out_w;\
		size_t in_h = out_h;\
		size_t in_map = out_map;\
		printf("%s %zdx%zd fmap:%zd->", #class_name, in_w, in_h, in_map);\
		printf(" %zdx%zd fmap:%zd\n", out_w, out_h, out_map);\
		return  tiny_dnn::class_name();\
	}

#define ACTIVATIN_FUNC2(class_name, param) \
	{\
		size_t in_w = out_w;\
		size_t in_h = out_h;\
		size_t in_map = out_map;\
		printf("%s %zdx%zd fmap:%zd->", #class_name, in_w, in_h, in_map);\
		printf(" %zdx%zd fmap:%zd\n", out_w, out_h, out_map);\
		return  tiny_dnn::class_name(param);\
	}
	
	inline  tiny_dnn::selu_layer selu_layer() { ACTIVATIN_FUNC(selu_layer) }
	inline  tiny_dnn::relu_layer relu() { ACTIVATIN_FUNC(relu_layer) }
	inline  tiny_dnn::leaky_relu_layer leaky_relu() { ACTIVATIN_FUNC2(leaky_relu_layer, float_t(0.0001)) }
	inline  tiny_dnn::elu_layer elu() { ACTIVATIN_FUNC(elu_layer) }
	inline  tiny_dnn::tanh_layer tanh() { ACTIVATIN_FUNC(tanh_layer) }
	inline  tiny_dnn::sigmoid_layer sigmoid() { ACTIVATIN_FUNC(sigmoid_layer) }
	inline  tiny_dnn::softmax_layer softmax(int classnum) { ACTIVATIN_FUNC2(softmax_layer, classnum) }
	inline  tiny_dnn::softplus_layer softplus() { ACTIVATIN_FUNC(softplus_layer) }
	inline  tiny_dnn::softsign_layer softsign() { ACTIVATIN_FUNC(softsign_layer) }


#if 0
	inline tiny_dnn::deconvolutional_layer add_decnv(
		size_t              out_channels = 1,
		size_t              window_size = 1,
		size_t              stride = 1,
		tiny_dnn::padding   pad_type = tiny_dnn::padding::valid,
		bool                has_bias = true	)
	{
		return add_decnv(out_channels, window_size, window_size, stride, stride, pad_type, has_bias);
	}
	inline tiny_dnn::deconvolutional_layer add_decnv(
		size_t              out_channels = 1,
		size_t              window_width = 1,
		size_t              window_height = 1,
		size_t              w_stride = 1,
		size_t              h_stride = 1,
		tiny_dnn::padding   pad_type = tiny_dnn::padding::valid,
		bool                has_bias = true	)
	{
		size_t in_w = out_w;
		size_t in_h = out_h;
		size_t in_map = out_map;
		out_map = out_channels;

		size_t w_dilation = 1;
		size_t h_dilation = 1;

		tiny_dnn::deconvolutional_layer layer = tiny_dnn::deconvolutional_layer(in_w, in_h, window_width, window_height, in_map, out_map, pad_type, has_bias, w_stride, h_stride, backend_type);
		out_w = conv_out_length(in_w, window_width, w_stride, w_dilation, pad_type);
		out_h = conv_out_length(in_h, window_height, h_stride, h_dilation, pad_type);

		//out_w = deconv_out_length(in_w, window_width, w_stride);
		//out_h = deconv_out_length(in_h, window_height, h_stride);
		out_w = deconv_out_unpadded_length(in_w, window_width, w_stride, pad_type);
		out_h = deconv_out_unpadded_length(in_h, window_height, h_stride, pad_type);
		//out_w = layer.out_shape()[0].size();
		//out_h = layer.out_shape()[1].size();

		printf("deconvolutional_layer %zdx%zd filter(%zd,%zd) stride(%zd,%zd) fmap:%zd->", in_w, in_h, window_width, window_height, w_stride, h_stride, in_map);
		printf(" %zdx%zd fmap:%zd\n", out_w, out_h, out_map);
		PARAMETER_NUM(layer);

		return layer;
	}
#endif



	inline tiny_dnn::convolutional_layer add_cnv(
		size_t              out_channels = 1,
		size_t              window_size = 1,
		size_t              stride = 1,
		tiny_dnn::padding   pad_type = tiny_dnn::padding::valid,
		bool                has_bias = true,
		tiny_dnn::core::connection_table &connection_table = tiny_dnn::core::connection_table()
	)
	{
		return add_cnv(out_channels, window_size, window_size, stride, stride, pad_type, has_bias, connection_table);
	}

	inline tiny_dnn::convolutional_layer add_cnv(
		size_t              out_channels = 1,
		size_t              window_width = 1,
		size_t              window_height = 1,
		size_t              w_stride = 1,
		size_t              h_stride = 1,
		tiny_dnn::padding   pad_type = tiny_dnn::padding::valid,
		bool                has_bias = true,
		tiny_dnn::core::connection_table &connection_table = tiny_dnn::core::connection_table()
	)
	{
		size_t in_w = out_w;
		size_t in_h = out_h;
		size_t in_map = out_map;
		out_map = out_channels;

		size_t w_dilation = 1;
		size_t h_dilation = 1;

		tiny_dnn::convolutional_layer layer = tiny_dnn::convolutional_layer(in_w, in_h, window_width, window_height, in_map, out_map, pad_type, has_bias, w_stride, h_stride, w_dilation, h_dilation, backend_type);
		out_w = conv_out_length(in_w, window_width, w_stride, w_dilation, pad_type);
		out_h = conv_out_length(in_h, window_height, h_stride, h_dilation, pad_type);
		//if (out_h <= 0 ) out_h = 1;

		printf("convolutional_layer %zdx%zd filter(%zd,%zd) stride(%zd,%zd) fmap:%zd->", in_w, in_h, window_width, window_height, w_stride, h_stride, in_map);
		printf(" %zdx%zd fmap:%zd\n", out_w, out_h, out_map);
		PARAMETER_NUM(layer);

		return layer;
	}

	inline tiny_dnn::max_pooling_layer add_maxpool(
		size_t pooling_size,
		padding    pad_type = padding::valid)

	{
		return add_maxpool(pooling_size, pooling_size, pooling_size, pooling_size, pad_type);
	}
	inline tiny_dnn::max_pooling_layer add_maxpool(
		size_t pooling_size,
		size_t stride,
		padding    pad_type = padding::valid)

	{
		return add_maxpool(pooling_size, pooling_size, stride, stride, pad_type);
	}

	inline tiny_dnn::max_pooling_layer add_maxpool(
		size_t pooling_size_x,
		size_t pooling_size_y,
		size_t stride_x,
		size_t stride_y,
		padding    pad_type = padding::valid)

	{
		size_t in_w = out_w;
		size_t in_h = out_h;
		size_t in_map = out_map;
		bool ceil_mode = false;
		size_t dilation = 1;

		tiny_dnn::max_pooling_layer layer = tiny_dnn::max_pooling_layer(in_w, in_h, in_map, pooling_size_x, pooling_size_y, stride_x, stride_y, ceil_mode, pad_type, backend_type);

		out_w = conv_out_length(in_w, pooling_size_x, stride_x, dilation, pad_type);
		out_h = conv_out_length(in_h, pooling_size_y, stride_y, dilation, pad_type);
		//if (out_h <= 0) out_h = 1;

		printf("max_pooling_layer %zdx%zd filter(%zd,%zd) stride(%zd,%zd) fmap:%zd->", in_w, in_h, pooling_size_x, pooling_size_y, stride_x, stride_y, in_map);
		printf(" %zdx%zd fmap:%zd\n", out_w, out_h, out_map);
		return layer;
	}

	inline tiny_dnn::average_pooling_layer add_avepool(
		size_t pooling_size,
		padding    pad_type = padding::valid)

	{
		return add_avepool(pooling_size, pooling_size, pooling_size, pooling_size, pad_type);
	}
	inline tiny_dnn::average_pooling_layer add_avepool(
		size_t pooling_size,
		size_t stride,
		padding    pad_type = padding::valid)

	{
		return add_avepool(pooling_size, pooling_size, stride, stride, pad_type);
	}
	inline tiny_dnn::average_pooling_layer add_avepool(
		size_t pooling_size_x,
		size_t pooling_size_y,
		size_t stride_x,
		size_t stride_y,
		padding    pad_type = padding::valid)

	{
		size_t in_w = out_w;
		size_t in_h = out_h;
		size_t in_map = out_map;
		size_t dilation = 1;
		bool ceil_mode = false;

		tiny_dnn::average_pooling_layer layer = tiny_dnn::average_pooling_layer(in_w, in_h, in_map, pooling_size_x, pooling_size_y, stride_x, stride_y, ceil_mode, pad_type);
		out_w = conv_out_length(in_w, pooling_size_x, stride_x, dilation, pad_type);
		out_h = conv_out_length(in_h, pooling_size_y, stride_y, dilation, pad_type);

		printf("average_pooling_layer %zdx%zd filter(%zd,%zd) stride(%zd,%zd) fmap:%zd->", in_w, in_h, pooling_size_x, pooling_size_y, stride_x, stride_y, in_map);
		printf(" %zdx%zd fmap:%zd\n", out_w, out_h, out_map);
		return layer;
	}

	inline tiny_dnn::batch_normalization_layer add_batnorm(
		tiny_dnn::layer &prev_layer,
		float_t epsilon = 1e-5,
		float_t momentum = 0.999,
		tiny_dnn::net_phase phase = tiny_dnn::net_phase::train
	)
	{
		size_t in_w = out_w;
		size_t in_h = out_h;
		size_t in_map = out_map;

		tiny_dnn::batch_normalization_layer layer = tiny_dnn::batch_normalization_layer(prev_layer, epsilon, momentum, phase);

		printf("batch_normalization_layer %zdx%zd  fmap:%zd->", in_w, in_h, in_map);
		printf(" %zdx%zd fmap:%zd\n", out_w, out_h, out_map);
		PARAMETER_NUM(layer);
		return layer;

	}
	inline tiny_dnn::batch_normalization_layer add_batnorm(
		float_t epsilon = 1e-5,
		float_t momentum = 0.999,
		tiny_dnn::net_phase phase = tiny_dnn::net_phase::train
	)
	{
		size_t in_w = out_w;
		size_t in_h = out_h;
		size_t in_map = out_map;

		tiny_dnn::batch_normalization_layer layer = tiny_dnn::batch_normalization_layer(in_w*in_h, in_map, epsilon, momentum, phase);

		printf("batch_normalization_layer %zdx%zd  fmap:%zd->", in_w, in_h, in_map);
		printf(" %zdx%zd fmap:%zd\n", out_w, out_h, out_map);
		PARAMETER_NUM(layer);
		return layer;

	}


	inline tiny_dnn::dropout_layer add_dropout(
		float_t dropout_rate,
		tiny_dnn::net_phase phase = tiny_dnn::net_phase::train
	)
	{
		size_t in_w = out_w;
		size_t in_h = out_h;
		size_t in_map = out_map;

		size_t in_dim = in_w*in_h*in_map;

		tiny_dnn::dropout_layer layer = tiny_dnn::dropout_layer(in_dim, dropout_rate, phase);

		printf("dropout_layer %.3f %zdx%zd fmap:%zd->", dropout_rate, in_w, in_h, in_map);
		printf(" %zdx%d fmap:%d\n", in_dim, 1, 1);
		return layer;

	}

	inline tiny_dnn::linear_layer add_linear(
		size_t out_dim,
		float_t scale = float_t{ 1 },
		float_t bias = float_t{ 0 }
	)
	{
		size_t in_w = out_w;
		size_t in_h = out_h;
		size_t in_map = out_map;
		size_t in_dim = in_w*in_h*in_map;

		tiny_dnn::linear_layer layer = tiny_dnn::linear_layer(out_dim, scale, bias);

		out_w = out_dim;
		out_h = 1;
		out_map = 1;

		printf("linear_layer %zdx%zd fmap:%zd->", in_w, in_h, in_map);
		printf(" %zdx%d fmap:%d\n", out_dim, 1, 1);
		PARAMETER_NUM(layer);
		return layer;
	}

	inline tiny_dnn::input_layer add_input(
		size_t out_dim
	)
	{
		size_t in_w = out_w;
		size_t in_h = out_h;
		size_t in_map = out_map;
		size_t in_dim = in_w*in_h*in_map;

		tiny_dnn::input_layer layer = tiny_dnn::input_layer(in_dim);

		out_w = out_dim;
		out_h = 1;
		out_map = 1;

		printf("input_layer %zdx%zd fmap:%zd->", in_w, in_h, in_map);
		printf(" %zdx%d fmap:%d\n", out_dim, 1, 1);
		PARAMETER_NUM(layer);
		return layer;
	}

	inline tiny_dnn::fully_connected_layer add_fc(
		size_t out_dim,
		bool       has_bias = true
	)
	{
		size_t in_w = out_w;
		size_t in_h = out_h;
		size_t in_map = out_map;
		size_t in_dim = in_w*in_h*in_map;

		tiny_dnn::fully_connected_layer layer = tiny_dnn::fully_connected_layer(in_dim, out_dim, has_bias, backend_type);

		out_w = out_dim;
		out_h = 1;
		out_map = 1;

		printf("fully_connected_layer %zdx%zd fmap:%zd->", in_w, in_h, in_map);
		printf(" %zdx%d fmap:%d\n", out_dim, 1, 1);
		PARAMETER_NUM(layer);
		return layer;
	}
	inline tiny_dnn::recurrent_layer add_rnn(
		std::string& rnn_type,
		size_t hidden_size,
		int seq_len,
		const recurrent_params& prmam
	)
	{
		size_t in_w = out_w;
		size_t in_h = out_h;
		size_t in_map = out_map;
		size_t input_size = in_w*in_h*in_map;

		out_w = hidden_size;
		out_h = 1;
		out_map = 1;

		printf("recurrent_layer_%s %zdx%zd fmap:%zd->", rnn_type.c_str(), in_w, in_h, in_map);
		printf(" %zdx%d fmap:%d\n", hidden_size, 1, 1);

		if (rnn_type == "rnn") {
			tiny_dnn::recurrent_layer layer = recurrent(tiny_dnn::rnn(input_size, hidden_size), seq_len, prmam);
			PARAMETER_NUM(layer);
			return layer;
		}
		else if (rnn_type == "gru") {
			tiny_dnn::recurrent_layer layer = recurrent(tiny_dnn::gru(input_size, hidden_size), seq_len, prmam);
			PARAMETER_NUM(layer);
			return layer;
		}
		else if (rnn_type == "lstm") {
			tiny_dnn::recurrent_layer layer = recurrent(tiny_dnn::lstm(input_size, hidden_size), seq_len, prmam);
			PARAMETER_NUM(layer);
			return layer;
		}
		else
		{
			tiny_dnn::recurrent_layer layer = recurrent(tiny_dnn::rnn(input_size, hidden_size), seq_len, prmam);
			PARAMETER_NUM(layer);
			return layer;
		}
	}
};

#undef ACTIVATIN_SYMBL
#undef ACTIVATIN_FUNC

#endif