#include "trt_gpt.h"

#include <NvInfer.h>
#include <NvInferVersion.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#include <process.h>
#endif

namespace {

using namespace nvinfer1;

class TrtLogger final : public ILogger {
 public:
  void log(Severity severity, char const* message) noexcept override {
    if (severity <= Severity::kWARNING) {
      std::fprintf(stderr, "[TensorRT] %s\n", message ? message : "");
    }
  }
};

TrtLogger g_logger;

[[noreturn]] void fail(std::string const& message) {
  throw std::runtime_error(message);
}

void check_cuda(cudaError_t status, char const* operation) {
  if (status != cudaSuccess) {
    fail(std::string(operation) + ": " + cudaGetErrorString(status));
  }
}

void check_trt(bool status, char const* operation) {
  if (!status) {
    fail(std::string("TensorRT ") + operation + " failed");
  }
}

std::filesystem::path path_from_utf8(char const* value) {
  if (!value || !*value) fail("empty TensorRT model or cache path");
  return std::filesystem::u8path(value);
}

std::string parser_errors(nvonnxparser::IParser& parser) {
  std::ostringstream message;
  for (int32_t i = 0; i < parser.getNbErrors(); ++i) {
    auto const* error = parser.getError(i);
    if (error) message << "\n  " << error->desc();
  }
  return message.str();
}

struct EngineContext {
  std::unique_ptr<IRuntime> runtime;
  std::unique_ptr<ICudaEngine> engine;
  std::unique_ptr<IExecutionContext> context;
};

class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(size_t bytes) { allocate(bytes); }
  DeviceBuffer(DeviceBuffer const&) = delete;
  DeviceBuffer& operator=(DeviceBuffer const&) = delete;
  DeviceBuffer(DeviceBuffer&& other) noexcept
      : pointer_(other.pointer_), bytes_(other.bytes_) {
    other.pointer_ = nullptr;
    other.bytes_ = 0;
  }
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      reset();
      pointer_ = other.pointer_;
      bytes_ = other.bytes_;
      other.pointer_ = nullptr;
      other.bytes_ = 0;
    }
    return *this;
  }
  ~DeviceBuffer() { reset(); }

  void allocate(size_t bytes) {
    reset();
    if (bytes == 0) fail("TensorRT device buffer cannot be empty");
    bytes_ = bytes;
    check_cuda(cudaMalloc(&pointer_, bytes_), "cudaMalloc");
  }
  void reset() noexcept {
    if (pointer_) cudaFree(pointer_);
    pointer_ = nullptr;
    bytes_ = 0;
  }
  void* get() const noexcept { return pointer_; }
  size_t size() const noexcept { return bytes_; }

 private:
  void* pointer_{nullptr};
  size_t bytes_{0};
};

class PinnedBuffer {
 public:
  PinnedBuffer() = default;
  explicit PinnedBuffer(size_t bytes) { allocate(bytes); }
  PinnedBuffer(PinnedBuffer const&) = delete;
  PinnedBuffer& operator=(PinnedBuffer const&) = delete;
  ~PinnedBuffer() {
    if (pointer_) cudaFreeHost(pointer_);
  }
  void allocate(size_t bytes) {
    if (pointer_) cudaFreeHost(pointer_);
    if (bytes == 0) fail("TensorRT pinned buffer cannot be empty");
    bytes_ = bytes;
    check_cuda(cudaHostAlloc(&pointer_, bytes_, cudaHostAllocPortable),
               "cudaHostAlloc");
  }
  void* get() const noexcept { return pointer_; }
  size_t size() const noexcept { return bytes_; }

 private:
  void* pointer_{nullptr};
  size_t bytes_{0};
};

class CudaStream {
 public:
  CudaStream() {
    check_cuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
               "cudaStreamCreateWithFlags");
  }
  CudaStream(CudaStream const&) = delete;
  CudaStream& operator=(CudaStream const&) = delete;
  ~CudaStream() {
    if (stream_) cudaStreamDestroy(stream_);
  }
  cudaStream_t get() const noexcept { return stream_; }

 private:
  cudaStream_t stream_{nullptr};
};

size_t element_size(DataType type) {
  switch (type) {
    case DataType::kFLOAT:
      return sizeof(float);
    case DataType::kINT64:
      return sizeof(int64_t);
    default:
      fail("GPT TensorRT graph expected only FP32 and INT64 tensors");
  }
}

size_t tensor_elements(Dims const& dims, char const* name) {
  size_t count = 1;
  for (int32_t axis = 0; axis < dims.nbDims; ++axis) {
    if (dims.d[axis] <= 0) {
      fail(std::string("TensorRT produced a non-static shape for ") + name);
    }
    count *= static_cast<size_t>(dims.d[axis]);
  }
  return count;
}

size_t tensor_bytes(IExecutionContext& context, ICudaEngine& engine,
                    char const* name) {
  return tensor_elements(context.getTensorShape(name), name) *
         element_size(engine.getTensorDataType(name));
}

void set_address(IExecutionContext& context, char const* name, void* pointer) {
  check_trt(context.setTensorAddress(name, pointer), "setTensorAddress");
}

bool has_dynamic_dimension(Dims const& dimensions) {
  for (int32_t axis = 0; axis < dimensions.nbDims; ++axis) {
    if (dimensions.d[axis] < 0) return true;
  }
  return false;
}

void set_static_input_shape(EngineContext& engine, char const* name,
                            Dims const& shape) {
  Dims const declared = engine.engine->getTensorShape(name);
  if (has_dynamic_dimension(declared)) {
    check_trt(engine.context->setInputShape(name, shape), "setInputShape");
    return;
  }
  if (declared.nbDims != shape.nbDims) {
    fail(std::string("TensorRT input rank mismatch for ") + name);
  }
  for (int32_t axis = 0; axis < shape.nbDims; ++axis) {
    if (declared.d[axis] != shape.d[axis]) {
      fail(std::string("TensorRT input shape mismatch for ") + name);
    }
  }
}

Dims make_profile_dims(Dims source, bool minimum) {
  for (int32_t axis = 0; axis < source.nbDims; ++axis) {
    if (source.d[axis] < 0) {
      if (axis == source.nbDims - 1) {
        source.d[axis] = minimum ? 1 : 128;
      } else {
        source.d[axis] = 1;
      }
    }
  }
  return source;
}

Dims make_profile_max_dims(Dims source) {
  for (int32_t axis = 0; axis < source.nbDims; ++axis) {
    if (source.d[axis] < 0) {
      // ABI-1 v2Pro rejects requests whose text, reference and semantic budget
      // exceed 512. Dynamic sequence axes therefore use this exact upper bound.
      source.d[axis] = axis == source.nbDims - 1 ? 512 : 1;
    }
  }
  return source;
}

EngineContext load_or_build_engine(std::filesystem::path const& onnx_path,
                                   std::filesystem::path const& plan_path) {
  EngineContext result;
  result.runtime.reset(createInferRuntime(g_logger));
  if (!result.runtime) fail("TensorRT runtime creation failed");

  if (std::filesystem::exists(plan_path)) {
    std::ifstream input(plan_path, std::ios::binary | std::ios::ate);
    if (input) {
      auto const length = input.tellg();
      if (length > 0) {
        std::vector<char> bytes(static_cast<size_t>(length));
        input.seekg(0);
        input.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        if (input) {
          result.engine.reset(result.runtime->deserializeCudaEngine(
              bytes.data(), bytes.size()));
        }
      }
    }
    if (!result.engine) {
      std::fprintf(stderr,
                   "[TensorRT] cached plan could not be loaded; rebuilding %s\n",
                   onnx_path.string().c_str());
    }
  }

  if (!result.engine) {
    std::unique_ptr<IBuilder> builder(createInferBuilder(g_logger));
    if (!builder) fail("TensorRT builder creation failed");
    // TensorRT 10+ uses explicit batch dimensions unconditionally; passing the
    // removed kEXPLICIT_BATCH flag is not valid in TensorRT 11.
    std::unique_ptr<INetworkDefinition> network(builder->createNetworkV2(0U));
    if (!network) fail("TensorRT network creation failed");
    std::unique_ptr<nvonnxparser::IParser> parser(
        nvonnxparser::createParser(*network, g_logger));
    if (!parser) fail("TensorRT ONNX parser creation failed");
    if (!parser->parseFromFile(onnx_path.string().c_str(),
                              static_cast<int32_t>(ILogger::Severity::kWARNING))) {
      fail("TensorRT could not parse ONNX graph " + onnx_path.string() +
           parser_errors(*parser));
    }

    std::unique_ptr<IBuilderConfig> config(builder->createBuilderConfig());
    if (!config) fail("TensorRT builder config creation failed");
    config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE,
                               4ULL * 1024ULL * 1024ULL * 1024ULL);
    config->clearFlag(BuilderFlag::kTF32);
    config->setMaxAuxStreams(0);

    bool dynamic_inputs = false;
    for (int32_t index = 0; index < network->getNbInputs(); ++index) {
      dynamic_inputs |= has_dynamic_dimension(network->getInput(index)->getDimensions()) ||
                        network->getInput(index)->isShapeTensor();
    }
    if (dynamic_inputs) {
      IOptimizationProfile* profile = builder->createOptimizationProfile();
      if (!profile) fail("TensorRT optimization profile creation failed");
      for (int32_t index = 0; index < network->getNbInputs(); ++index) {
      auto* input = network->getInput(index);
      Dims const source = input->getDimensions();
      if (input->isShapeTensor()) {
        if (std::strcmp(input->getName(), "active_len") != 0) {
          fail("unexpected native GPT shape input");
        }
        int64_t const minimum = 1, optimum = 256, maximum = 512;
        if (!profile->setShapeValuesV2(input->getName(), OptProfileSelector::kMIN, &minimum, 1) ||
            !profile->setShapeValuesV2(input->getName(), OptProfileSelector::kOPT, &optimum, 1) ||
            !profile->setShapeValuesV2(input->getName(), OptProfileSelector::kMAX, &maximum, 1)) {
          fail("could not configure active_len shape profile");
        }
      }
      if (!has_dynamic_dimension(source)) continue;
      Dims const minimum = make_profile_dims(source, true);
      Dims const optimum = make_profile_dims(source, false);
      Dims const maximum = make_profile_max_dims(source);
      char const* name = input->getName();
        if (!profile->setDimensions(name, OptProfileSelector::kMIN, minimum) ||
            !profile->setDimensions(name, OptProfileSelector::kOPT, optimum) ||
            !profile->setDimensions(name, OptProfileSelector::kMAX, maximum)) {
          fail(std::string("TensorRT rejected the ABI-1 dynamic profile for ") +
               name);
        }
      }
      int32_t const profile_index = config->addOptimizationProfile(profile);
      if (profile_index < 0) fail("TensorRT could not add the dynamic profile");
    }

    std::unique_ptr<IHostMemory> serialized(
        builder->buildSerializedNetwork(*network, *config));
    if (!serialized) {
      fail("TensorRT engine build failed for " + onnx_path.string());
    }
    std::filesystem::create_directories(plan_path.parent_path());
    std::filesystem::path temporary = plan_path;
#ifdef _WIN32
    temporary += ".tmp." + std::to_string(_getpid());
#else
    temporary += ".tmp";
#endif
    {
      std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
      if (!output) fail("could not create TensorRT engine cache file");
      output.write(static_cast<char const*>(serialized->data()),
                   static_cast<std::streamsize>(serialized->size()));
      output.flush();
      if (!output) fail("could not write TensorRT engine cache file");
    }
    std::error_code rename_error;
    std::filesystem::rename(temporary, plan_path, rename_error);
    if (rename_error) {
      std::error_code remove_error;
      std::filesystem::remove(plan_path, remove_error);
      if (!remove_error) {
        std::filesystem::rename(temporary, plan_path);
      } else {
        std::filesystem::remove(temporary);
      }
    }
    result.engine.reset(result.runtime->deserializeCudaEngine(
        serialized->data(), serialized->size()));
  }

  if (!result.engine) fail("TensorRT engine deserialization failed");
  result.context.reset(result.engine->createExecutionContext());
  if (!result.context) fail("TensorRT execution context creation failed");
  return result;
}

class GptRunner {
 public:
  GptRunner(std::filesystem::path encoder_onnx,
            std::filesystem::path encoder_plan,
            std::filesystem::path step_onnx,
            std::filesystem::path step_plan) {
    check_cuda(cudaSetDevice(0), "cudaSetDevice");
    encoder_ = load_or_build_engine(encoder_onnx, encoder_plan);
    step_ = load_or_build_engine(step_onnx, step_plan);
    stream_ = std::make_unique<CudaStream>();
    prepare_encoder_context();
    prepare_step_context();
    allocate_buffers();
    for (int32_t i = 0; i < step_.engine->getNbIOTensors(); ++i) {
      if (std::strcmp(step_.engine->getIOTensorName(i), "active_len") == 0) {
        if (step_.engine->getTensorLocation("active_len") != TensorLocation::kHOST ||
            !step_.engine->isShapeInferenceIO("active_len") ||
            step_.engine->getTensorDataType("active_len") != DataType::kINT64 ||
            tensor_elements(step_.engine->getTensorShape("active_len"), "active_len") != 1) {
          fail("active_len must be a host INT64[1] shape input");
        }
        host_active_len_.allocate(sizeof(int64_t));
        active_len_input_ = true;
        set_address(*step_.context, "active_len", host_active_len_.get());
      }
    }
  }

  void run_encoder(int64_t const* phoneme_ids, size_t phoneme_count,
                   int64_t const* prompt_ids, size_t prompt_count,
                   float const* bert_features, size_t bert_feature_count,
                   float* topk_values, int64_t* topk_indices,
                   size_t topk_capacity, size_t* topk_count,
                   int64_t* x_length, int64_t* y_length) {
    if (!phoneme_ids || !prompt_ids || !bert_features || !topk_values ||
        !topk_indices || !topk_count || !x_length || !y_length) {
      fail("GPT Encoder received a null input or output pointer");
    }
    if (phoneme_count == 0 || phoneme_count > kMaxSequence || prompt_count == 0 ||
        prompt_count > kMaxSequence ||
        bert_feature_count != phoneme_count * kBertWidth) {
      fail("GPT Encoder inputs exceed the ABI-1 shape contract");
    }
    check_cuda(cudaSetDevice(0), "cudaSetDevice");
    std::memcpy(host_phonemes_.get(), phoneme_ids,
                phoneme_count * sizeof(int64_t));
    std::memcpy(host_prompts_.get(), prompt_ids, prompt_count * sizeof(int64_t));
    std::memcpy(host_bert_.get(), bert_features,
                bert_feature_count * sizeof(float));

    Dims2 const phone_shape{1, static_cast<int32_t>(phoneme_count)};
    Dims2 const prompt_shape{1, static_cast<int32_t>(prompt_count)};
    Dims3 const feature_shape{1, static_cast<int32_t>(kBertWidth),
                              static_cast<int32_t>(phoneme_count)};
    set_input_shape(encoder_, "phoneme_ids", phone_shape);
    set_input_shape(encoder_, "prompts", prompt_shape);
    set_input_shape(encoder_, "bert_feature", feature_shape);

    check_cuda(cudaMemcpyAsync(device_phonemes_.get(), host_phonemes_.get(),
                               phoneme_count * sizeof(int64_t),
                               cudaMemcpyHostToDevice, stream_->get()),
               "copy phoneme IDs to device");
    check_cuda(cudaMemcpyAsync(device_prompts_.get(), host_prompts_.get(),
                               prompt_count * sizeof(int64_t),
                               cudaMemcpyHostToDevice, stream_->get()),
               "copy prompt IDs to device");
    check_cuda(cudaMemcpyAsync(device_bert_.get(), host_bert_.get(),
                               bert_feature_count * sizeof(float),
                               cudaMemcpyHostToDevice, stream_->get()),
               "copy BERT features to device");
    if (device_x_length_.size() != sizeof(int64_t) ||
        device_y_length_.size() != sizeof(int64_t)) {
      fail("GPT Encoder lengths must be scalar INT64 tensors");
    }

    set_address(*encoder_.context, "phoneme_ids", device_phonemes_.get());
    set_address(*encoder_.context, "prompts", device_prompts_.get());
    set_address(*encoder_.context, "bert_feature", device_bert_.get());
    set_address(*encoder_.context, "topk_values", device_encoder_values_.get());
    set_address(*encoder_.context, "topk_indices", device_encoder_indices_.get());
    set_address(*encoder_.context, "x_len", device_x_length_.get());
    set_address(*encoder_.context, "y_len", device_y_length_.get());
    set_address(*encoder_.context, "k_cache", cache_key_[0].get());
    set_address(*encoder_.context, "v_cache", cache_value_[0].get());
    check_trt(encoder_.context->enqueueV3(stream_->get()),
              "GPT Encoder enqueueV3");
    *topk_count = enqueue_sampling_outputs(
        *encoder_.context, *encoder_.engine, device_encoder_values_,
        device_encoder_indices_, topk_capacity, host_encoder_values_,
        host_encoder_indices_, stream_->get());
    check_cuda(cudaMemcpyAsync(host_x_length_.get(), device_x_length_.get(),
                               sizeof(int64_t), cudaMemcpyDeviceToHost,
                               stream_->get()),
               "copy x_len to host");
    check_cuda(cudaMemcpyAsync(host_y_length_.get(), device_y_length_.get(),
                               sizeof(int64_t), cudaMemcpyDeviceToHost,
                               stream_->get()),
               "copy y_len to host");
    check_cuda(cudaStreamSynchronize(stream_->get()), "synchronize GPT Encoder");
    copy_sampling_to_host(*topk_count, topk_values, topk_indices,
                          host_encoder_values_, host_encoder_indices_);
    *x_length = *static_cast<int64_t*>(host_x_length_.get());
    *y_length = *static_cast<int64_t*>(host_y_length_.get());
    current_cache_ = 0;
    cache_initialized_ = true;
  }

  void run_step(int64_t sample, int64_t token_index, float* topk_values,
                int64_t* topk_indices, size_t topk_capacity,
                size_t* topk_count) {
    if (!cache_initialized_) {
      fail("GPT Step was called before the current request's GPT Encoder");
    }
    if (!topk_values || !topk_indices || !topk_count) {
      fail("GPT Step received a null output pointer");
    }
    if (token_index < 0 || token_index >= 512) {
      fail("GPT Step token index is outside the ABI-1 cache");
    }
    if (active_len_input_) {
      int64_t const x_length = *static_cast<int64_t*>(host_x_length_.get());
      int64_t const y_length = *static_cast<int64_t*>(host_y_length_.get());
      if (x_length < 0 || y_length < 0 || x_length > static_cast<int64_t>(kMaxSequence) ||
          y_length > static_cast<int64_t>(kMaxSequence)) {
        fail("native GPT encoder lengths are outside the ABI-1 cache");
      }
      int64_t const active_len = x_length + y_length + token_index + 1;
      if (active_len < 1 || active_len > static_cast<int64_t>(kMaxSequence)) {
        fail("native GPT active_len is outside the ABI-1 cache");
      }
      *static_cast<int64_t*>(host_active_len_.get()) = active_len;
    }
    check_cuda(cudaSetDevice(0), "cudaSetDevice");
    *static_cast<int64_t*>(host_sample_.get()) = sample;
    *static_cast<int64_t*>(host_token_index_.get()) = token_index;
    check_cuda(cudaMemcpyAsync(device_sample_.get(), host_sample_.get(),
                               sizeof(int64_t), cudaMemcpyHostToDevice,
                               stream_->get()),
               "copy semantic sample to device");
    check_cuda(cudaMemcpyAsync(device_token_index_.get(),
                               host_token_index_.get(), sizeof(int64_t),
                               cudaMemcpyHostToDevice, stream_->get()),
               "copy token index to device");

    int32_t const next_cache = 1 - current_cache_;
    set_address(*step_.context, "samples", device_sample_.get());
    set_address(*step_.context, "idx", device_token_index_.get());
    set_address(*step_.context, "x_len", device_x_length_.get());
    set_address(*step_.context, "y_len", device_y_length_.get());
    set_address(*step_.context, "k_cache", cache_key_[current_cache_].get());
    set_address(*step_.context, "v_cache", cache_value_[current_cache_].get());
    set_address(*step_.context, "topk_values", device_step_values_.get());
    set_address(*step_.context, "topk_indices", device_step_indices_.get());
    set_address(*step_.context, "k_cache_new", cache_key_[next_cache].get());
    set_address(*step_.context, "v_cache_new", cache_value_[next_cache].get());
    check_trt(step_.context->enqueueV3(stream_->get()),
              "GPT Step enqueueV3");
    *topk_count = enqueue_sampling_outputs(
        *step_.context, *step_.engine, device_step_values_, device_step_indices_,
        topk_capacity, host_step_values_, host_step_indices_, stream_->get());
    check_cuda(cudaStreamSynchronize(stream_->get()), "synchronize GPT Step");
    copy_sampling_to_host(*topk_count, topk_values, topk_indices,
                          host_step_values_, host_step_indices_);
    current_cache_ = next_cache;
  }

  void reset() noexcept {
    cache_initialized_ = false;
    current_cache_ = 0;
  }

 private:
  static constexpr size_t kBertWidth = 1024;
  static constexpr size_t kMaxSequence = 512;

  static void set_input_shape(EngineContext& engine, char const* name,
                              Dims const& shape) {
    Dims const declared = engine.engine->getTensorShape(name);
    if (has_dynamic_dimension(declared)) {
      check_trt(engine.context->setInputShape(name, shape),
                "setInputShape");
    } else {
      for (int32_t axis = 0; axis < shape.nbDims; ++axis) {
        if (declared.nbDims != shape.nbDims ||
            declared.d[axis] != shape.d[axis]) {
          fail(std::string("GPT Encoder shape mismatch for ") + name);
        }
      }
    }
  }

  static size_t enqueue_sampling_outputs(IExecutionContext& context,
                                        ICudaEngine& engine,
                                        DeviceBuffer& device_values,
                                        DeviceBuffer& device_indices,
                                        size_t capacity,
                                        PinnedBuffer& host_values,
                                        PinnedBuffer& host_indices,
                                        cudaStream_t stream) {
    size_t const values_count = tensor_elements(
        context.getTensorShape("topk_values"), "topk_values");
    size_t const indices_count = tensor_elements(
        context.getTensorShape("topk_indices"), "topk_indices");
    if (values_count == 0 || values_count != indices_count ||
        values_count > capacity) {
      fail("GPT top-k output shape is outside the native interface capacity");
    }
    if (engine.getTensorDataType("topk_values") != DataType::kFLOAT ||
        engine.getTensorDataType("topk_indices") != DataType::kINT64) {
      fail("GPT top-k output types must be FP32 values and INT64 indices");
    }
    check_cuda(cudaMemcpyAsync(host_values.get(), device_values.get(),
                               values_count * sizeof(float),
                               cudaMemcpyDeviceToHost, stream),
               "copy top-k values to host");
    check_cuda(cudaMemcpyAsync(host_indices.get(), device_indices.get(),
                               indices_count * sizeof(int64_t),
                               cudaMemcpyDeviceToHost, stream),
               "copy top-k indices to host");
    return values_count;
  }

  static void copy_sampling_to_host(size_t count, float* values_out,
                                    int64_t* indices_out,
                                    PinnedBuffer& host_values,
                                    PinnedBuffer& host_indices) {
    if (!values_out || !indices_out) fail("GPT sampling output pointer is null");
    std::memcpy(values_out, host_values.get(), count * sizeof(float));
    std::memcpy(indices_out, host_indices.get(),
                count * sizeof(int64_t));
  }

  void prepare_encoder_context() {
    Dims2 const default_sequence{1, 1};
    Dims3 const default_features{1, static_cast<int32_t>(kBertWidth), 1};
    set_input_shape(encoder_, "phoneme_ids", default_sequence);
    set_input_shape(encoder_, "prompts", default_sequence);
    set_input_shape(encoder_, "bert_feature", default_features);
  }

  void prepare_step_context() {
    for (int32_t index = 0; index < step_.engine->getNbIOTensors(); ++index) {
      char const* name = step_.engine->getIOTensorName(index);
      if (step_.engine->getTensorIOMode(name) != TensorIOMode::kINPUT) continue;
      if (std::strcmp(name, "samples") == 0) {
        set_static_input_shape(step_, name, Dims2{1, 1});
      } else if (std::strcmp(name, "x_len") == 0 ||
                 std::strcmp(name, "y_len") == 0 ||
                 std::strcmp(name, "idx") == 0) {
        Dims scalar_vector{};
        scalar_vector.nbDims = 1;
        scalar_vector.d[0] = 1;
        set_static_input_shape(step_, name, scalar_vector);
      } else if (has_dynamic_dimension(step_.engine->getTensorShape(name))) {
        fail(std::string("GPT Step has an unsupported dynamic input: ") +
             name);
      }
    }
  }

  void allocate_buffers() {
    device_phonemes_.allocate(kMaxSequence * sizeof(int64_t));
    device_prompts_.allocate(kMaxSequence * sizeof(int64_t));
    device_bert_.allocate(kBertWidth * kMaxSequence * sizeof(float));
    host_phonemes_.allocate(kMaxSequence * sizeof(int64_t));
    host_prompts_.allocate(kMaxSequence * sizeof(int64_t));
    host_bert_.allocate(kBertWidth * kMaxSequence * sizeof(float));

    size_t const enc_values = tensor_bytes(*encoder_.context, *encoder_.engine,
                                           "topk_values");
    size_t const enc_indices = tensor_bytes(*encoder_.context, *encoder_.engine,
                                            "topk_indices");
    size_t const step_values = tensor_bytes(*step_.context, *step_.engine,
                                            "topk_values");
    size_t const step_indices = tensor_bytes(*step_.context, *step_.engine,
                                             "topk_indices");
    if (enc_values != step_values || enc_indices != step_indices) {
      fail("GPT Encoder and Step top-k output shapes do not match");
    }
    if (encoder_.engine->getTensorDataType("phoneme_ids") != DataType::kINT64 ||
        encoder_.engine->getTensorDataType("prompts") != DataType::kINT64 ||
        encoder_.engine->getTensorDataType("bert_feature") != DataType::kFLOAT ||
        step_.engine->getTensorDataType("samples") != DataType::kINT64 ||
        step_.engine->getTensorDataType("idx") != DataType::kINT64 ||
        encoder_.engine->getTensorDataType("x_len") != DataType::kINT64 ||
        encoder_.engine->getTensorDataType("y_len") != DataType::kINT64 ||
        step_.engine->getTensorDataType("x_len") != DataType::kINT64 ||
        step_.engine->getTensorDataType("y_len") != DataType::kINT64) {
      fail("GPT TensorRT I/O type changed from the ABI-1 INT64/FP32 contract");
    }
    topk_count_ = enc_values / sizeof(float);
    if (enc_values % sizeof(float) != 0 || topk_count_ == 0 || topk_count_ > 256 ||
        step_indices % sizeof(int64_t) != 0) {
      fail("GPT top-k output shape is invalid");
    }
    device_encoder_values_.allocate(enc_values);
    device_encoder_indices_.allocate(enc_indices);
    device_step_values_.allocate(step_values);
    device_step_indices_.allocate(step_indices);
    host_encoder_values_.allocate(enc_values);
    host_encoder_indices_.allocate(enc_indices);
    host_step_values_.allocate(step_values);
    host_step_indices_.allocate(step_indices);
    host_x_length_.allocate(sizeof(int64_t));
    host_y_length_.allocate(sizeof(int64_t));
    device_x_length_.allocate(tensor_bytes(*encoder_.context, *encoder_.engine,
                                           "x_len"));
    device_y_length_.allocate(tensor_bytes(*encoder_.context, *encoder_.engine,
                                           "y_len"));
    if (tensor_bytes(*step_.context, *step_.engine, "x_len") !=
            device_x_length_.size() ||
        tensor_bytes(*step_.context, *step_.engine, "y_len") !=
            device_y_length_.size()) {
      fail("GPT Encoder and Step length tensor shapes do not match");
    }

    size_t const encoder_key =
        tensor_bytes(*encoder_.context, *encoder_.engine, "k_cache");
    size_t const encoder_value =
        tensor_bytes(*encoder_.context, *encoder_.engine, "v_cache");
    size_t const step_key =
        tensor_bytes(*step_.context, *step_.engine, "k_cache");
    size_t const step_value =
        tensor_bytes(*step_.context, *step_.engine, "v_cache");
    size_t const new_key =
        tensor_bytes(*step_.context, *step_.engine, "k_cache_new");
    size_t const new_value =
        tensor_bytes(*step_.context, *step_.engine, "v_cache_new");
    if (encoder_key != step_key || encoder_value != step_value ||
        step_key != new_key || step_value != new_value) {
      fail("GPT Encoder and Step cache buffer shapes do not match");
    }
    for (int index = 0; index < 2; ++index) {
      cache_key_[index].allocate(encoder_key);
      cache_value_[index].allocate(encoder_value);
    }

    device_sample_.allocate(sizeof(int64_t));
    device_token_index_.allocate(sizeof(int64_t));
    host_sample_.allocate(sizeof(int64_t));
    host_token_index_.allocate(sizeof(int64_t));
  }

  EngineContext encoder_;
  EngineContext step_;
  std::unique_ptr<CudaStream> stream_;
  DeviceBuffer device_phonemes_;
  DeviceBuffer device_prompts_;
  DeviceBuffer device_bert_;
  PinnedBuffer host_phonemes_;
  PinnedBuffer host_prompts_;
  PinnedBuffer host_bert_;
  DeviceBuffer device_encoder_values_;
  DeviceBuffer device_encoder_indices_;
  DeviceBuffer device_step_values_;
  DeviceBuffer device_step_indices_;
  PinnedBuffer host_encoder_values_;
  PinnedBuffer host_encoder_indices_;
  PinnedBuffer host_step_values_;
  PinnedBuffer host_step_indices_;
  DeviceBuffer device_x_length_;
  DeviceBuffer device_y_length_;
  PinnedBuffer host_x_length_;
  PinnedBuffer host_y_length_;
  DeviceBuffer device_sample_;
  DeviceBuffer device_token_index_;
  PinnedBuffer host_sample_;
  PinnedBuffer host_token_index_;
  std::array<DeviceBuffer, 2> cache_key_;
  std::array<DeviceBuffer, 2> cache_value_;
  size_t topk_count_{0};
  int32_t current_cache_{0};
  bool cache_initialized_{false};
  bool active_len_input_{false};
  PinnedBuffer host_active_len_;
};

void set_error(char* output, size_t capacity, std::string const& message) {
  if (!output || capacity == 0) return;
  size_t const count = std::min(capacity - 1, message.size());
  std::memcpy(output, message.data(), count);
  output[count] = '\0';
}

template <typename Function>
int32_t guard(Function&& function, char* error, size_t error_capacity) noexcept {
  try {
    function();
    if (error && error_capacity) error[0] = '\0';
    return 0;
  } catch (std::exception const& exception) {
    set_error(error, error_capacity, exception.what());
  } catch (...) {
    set_error(error, error_capacity, "unknown TensorRT native error");
  }
  return 1;
}

}  // namespace

extern "C" {

void* alice_trt_gpt_create(char const* encoder_onnx, char const* encoder_plan,
                           char const* step_onnx, char const* step_plan,
                           char* error, size_t error_capacity) {
  try {
    auto runner = std::make_unique<GptRunner>(
        path_from_utf8(encoder_onnx), path_from_utf8(encoder_plan),
        path_from_utf8(step_onnx), path_from_utf8(step_plan));
    if (error && error_capacity) error[0] = '\0';
    return runner.release();
  } catch (std::exception const& exception) {
    set_error(error, error_capacity, exception.what());
  } catch (...) {
    set_error(error, error_capacity, "unknown TensorRT native creation error");
  }
  return nullptr;
}

int32_t alice_trt_gpt_run_encoder(
    void* handle, int64_t const* phoneme_ids, size_t phoneme_count,
    int64_t const* prompt_ids, size_t prompt_count, float const* bert_features,
    size_t bert_feature_count, float* topk_values, int64_t* topk_indices,
    size_t topk_capacity, size_t* topk_count, int64_t* x_length,
    int64_t* y_length, char* error, size_t error_capacity) {
  return guard(
      [&] {
        if (!handle) fail("TensorRT GPT handle is null");
        static_cast<GptRunner*>(handle)->run_encoder(
            phoneme_ids, phoneme_count, prompt_ids, prompt_count, bert_features,
            bert_feature_count, topk_values, topk_indices, topk_capacity,
            topk_count, x_length, y_length);
      },
      error, error_capacity);
}

int32_t alice_trt_gpt_run_step(void* handle, int64_t sample,
                               int64_t token_index, float* topk_values,
                               int64_t* topk_indices, size_t topk_capacity,
                               size_t* topk_count, char* error,
                               size_t error_capacity) {
  return guard(
      [&] {
        if (!handle) fail("TensorRT GPT handle is null");
        static_cast<GptRunner*>(handle)->run_step(
            sample, token_index, topk_values, topk_indices, topk_capacity,
            topk_count);
      },
      error, error_capacity);
}

void alice_trt_gpt_reset(void* handle) {
  if (handle) static_cast<GptRunner*>(handle)->reset();
}

void alice_trt_gpt_destroy(void* handle) {
  delete static_cast<GptRunner*>(handle);
}

}  // extern "C"
