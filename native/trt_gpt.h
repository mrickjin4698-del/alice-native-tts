#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef _WIN32
#define ALICE_TRT_EXPORT __declspec(dllexport)
#else
#define ALICE_TRT_EXPORT
#endif

extern "C" {

ALICE_TRT_EXPORT void* alice_trt_gpt_create(const char* encoder_onnx,
                                            const char* encoder_plan,
                                            const char* step_onnx,
                                            const char* step_plan,
                                            char* error,
                                            size_t error_capacity);

ALICE_TRT_EXPORT int32_t alice_trt_gpt_run_encoder(void* handle,
                                                   const int64_t* phoneme_ids,
                                                   size_t phoneme_count,
                                                   const int64_t* prompt_ids,
                                                   size_t prompt_count,
                                                   const float* bert_features,
                                                   size_t bert_feature_count,
                                                   float* topk_values,
                                                   int64_t* topk_indices,
                                                   size_t topk_capacity,
                                                   size_t* topk_count,
                                                   int64_t* x_length,
                                                   int64_t* y_length,
                                                   char* error,
                                                   size_t error_capacity);

ALICE_TRT_EXPORT int32_t alice_trt_gpt_run_step(void* handle,
                                                int64_t sample,
                                                int64_t token_index,
                                                float* topk_values,
                                                int64_t* topk_indices,
                                                size_t topk_capacity,
                                                size_t* topk_count,
                                                char* error,
                                                size_t error_capacity);

ALICE_TRT_EXPORT void alice_trt_gpt_reset(void* handle);
ALICE_TRT_EXPORT void alice_trt_gpt_destroy(void* handle);

struct AliceTrtInput {
  const char* name;
  const void* data;
  size_t bytes;
  int64_t shape[4];
  size_t rank;
};

struct AliceTrtOutput {
  const char* name;
  const float* data;
  size_t length;
  int64_t shape[4];
  size_t rank;
};

typedef bool (*AliceTrtCancelled)(void* user_data);

ALICE_TRT_EXPORT void* alice_trt_graph_load(const char* plan, char* error,
                                           size_t error_capacity);
ALICE_TRT_EXPORT int32_t alice_trt_graph_run(
    void* handle, const AliceTrtInput* inputs, size_t input_count,
    AliceTrtOutput* outputs, size_t output_count, size_t steps,
    AliceTrtCancelled cancelled, void* user_data, char* error,
    size_t error_capacity);
ALICE_TRT_EXPORT void alice_trt_graph_destroy(void* handle);

}
