#include <iostream>

#include "react-native-fast-opencv.h"
#include "jsi/Promise.h"
#include "jsi/TypedArray.h"
#include <FOCV_Ids.hpp>
#include <FOCV_Storage.hpp>
#include <FOCV_Function.hpp>
#include "FOCV_Object.hpp"
#include "ConvertImage.hpp"
#include "FOCV_JsiObject.hpp"
#include "opencv2/opencv.hpp"

using namespace mrousavy;

void OpenCVPlugin::installOpenCV(jsi::Runtime& runtime, std::shared_ptr<react::CallInvoker> callInvoker) {

    auto func = [=](jsi::Runtime& runtime,
                        const jsi::Value& thisArg,
                        const jsi::Value* args,
                        size_t count) -> jsi::Value {
        auto plugin = std::make_shared<OpenCVPlugin>(callInvoker);
        auto result = jsi::Object::createFromHostObject(runtime, plugin);

        return result;
    };

    auto jsiFunc = jsi::Function::createFromHostFunction(runtime,
                                                         jsi::PropNameID::forUtf8(runtime, "__loadOpenCV"),
                                                         1,
                                                         func);

    runtime.global().setProperty(runtime, "__loadOpenCV", jsiFunc);
    
}

OpenCVPlugin::OpenCVPlugin(std::shared_ptr<react::CallInvoker> callInvoker) : _callInvoker(callInvoker) {}

jsi::Value OpenCVPlugin::get(jsi::Runtime &runtime, const jsi::PropNameID &propNameId)
{
    auto propName = propNameId.utf8(runtime);

  if (propName == "frameBufferToMat") {
    return jsi::Function::createFromHostFunction(
        runtime, jsi::PropNameID::forAscii(runtime, "frameBufferToMat"), 1,
        [=](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* arguments,
            size_t count) -> jsi::Object {

        jsi::Object input = arguments[2].asObject(runtime);
        TypedArrayBase inputBuffer = getTypedArray(runtime, std::move(input));
        auto vec = inputBuffer.toVector(runtime);

        cv::Mat mat(arguments[0].asNumber(), arguments[1].asNumber(), CV_8UC3, vec.data());
        auto id = FOCV_Storage::save(mat);

        return FOCV_JsiObject::wrap(runtime, "mat", id);
    });
  }
  else if (propName == "bufferToMat") {
    return jsi::Function::createFromHostFunction(
        runtime, jsi::PropNameID::forAscii(runtime, "bufferToMat"), 1,
        [=](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* arguments,
            size_t count) -> jsi::Object {

        jsi::Object input = arguments[2].asObject(runtime);
        TypedArrayBase inputBuffer = getTypedArray(runtime, std::move(input));
        auto vec = inputBuffer.toVector(runtime);

        int rows = arguments[0].asNumber();
        int cols = arguments[1].asNumber();

        size_t bufferSize = vec.size();
        int channels = static_cast<int>(bufferSize / (rows * cols));

        int matType;
        if (channels == 1) {
            matType = CV_8UC1; // Grayscale
        } else if (channels == 3) {
            matType = CV_8UC3; // RGB
        } else if (channels == 4) {
            matType = CV_8UC4; // RGBA
        } else {
            throw std::runtime_error("Unsupported number of channels in buffer");
        }

        cv::Mat mat(rows, cols, matType, vec.data());
        auto id = FOCV_Storage::save(mat);

        return FOCV_JsiObject::wrap(runtime, "mat", id);
    });
  }
  else if (propName == "bufferF32ToMat") {
    return jsi::Function::createFromHostFunction(
        runtime, jsi::PropNameID::forAscii(runtime, "bufferF32ToMat"), 1,
        [=](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* arguments,
            size_t count) -> jsi::Object {

        jsi::Object input = arguments[2].asObject(runtime);
        TypedArrayBase inputBuffer = getTypedArray(runtime, std::move(input));
        auto vec = inputBuffer.toVector32F(runtime);

        int rows = arguments[0].asNumber();
        int cols = arguments[1].asNumber();

        size_t bufferSize = vec.size();
        int channels = static_cast<int>(bufferSize / (rows * cols));
              
        // Ensure the buffer size is consistent with rows, cols, and channels
        if (bufferSize != rows * cols * channels) {
            throw std::runtime_error("Buffer size mismatch with rows, cols, and channels.");
        }

        int matType;
        if (channels >= 1) {
            matType = CV_MAKETYPE(CV_32F, channels);  // CV_32F with N channels
        } else {
            throw std::runtime_error("Unsupported number of channels in buffer");
        }

        // Create cv::Mat and perform a deep copy
        cv::Mat mat(rows, cols, matType);
        memcpy(mat.data, vec.data(), bufferSize * sizeof(float));
        auto id = FOCV_Storage::save(mat);

        return FOCV_JsiObject::wrap(runtime, "mat", id);
    });
  }
  else if (propName == "base64ToMat") {
      return jsi::Function::createFromHostFunction(
          runtime, jsi::PropNameID::forAscii(runtime, "base64ToMat"), 1,
          [=](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* arguments,
              size_t count) -> jsi::Object {

          std::string base64 = arguments[0].asString(runtime).utf8(runtime);

                auto mat = ImageConverter::str2mat(base64);
                auto id = FOCV_Storage::save(mat);

                return FOCV_JsiObject::wrap(runtime, "mat", id);
            });
    }
  else if (propName == "matToBuffer") {
      return jsi::Function::createFromHostFunction(
          runtime, jsi::PropNameID::forAscii(runtime, "matToBuffer"), 1,
          [=](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* arguments,
              size_t count) -> jsi::Object {

                  std::string id = FOCV_JsiObject::id_from_wrap(runtime, arguments[0]);
                  auto mat = *FOCV_Storage::get<cv::Mat>(id);

                  jsi::Object value(runtime);

                  value.setProperty(runtime, "cols", jsi::Value(mat.cols));
                  value.setProperty(runtime, "rows", jsi::Value(mat.rows));
                  value.setProperty(runtime, "channels", jsi::Value(mat.channels()));

                  auto type = arguments[1].asString(runtime).utf8(runtime);
                  int size = mat.cols * mat.rows * mat.channels();

                  if(type == "uint8") {
                      auto arr = TypedArray<TypedArrayKind::Uint8Array>(runtime, size);
                      arr.updateUnsafe(runtime, (uint8_t*)mat.data, size * sizeof(uint8_t));
                      value.setProperty(runtime, "buffer", arr);
                  } else if(type == "float32") {
                      auto arr = TypedArray<TypedArrayKind::Float32Array>(runtime, size);
                      arr.updateUnsafe(runtime, (float*)mat.data, size * sizeof(float));
                      value.setProperty(runtime, "buffer", arr);
                  }

                  return value;
      });
    } else if (propName == "createObject") {
      return jsi::Function::createFromHostFunction(
          runtime, jsi::PropNameID::forAscii(runtime, "createObject"), 1,
          [=](jsi::Runtime& runtime, const jsi::Value& thisValue, const jsi::Value* arguments,
              size_t count) -> jsi::Object {

          return FOCV_Object::create(runtime, arguments);
      });
    }
    else if (propName == "toJSValue")
    {
        return jsi::Function::createFromHostFunction(
            runtime, jsi::PropNameID::forAscii(runtime, "toJSValue"), 1,
            [=](jsi::Runtime &runtime, const jsi::Value &thisValue, const jsi::Value *arguments,
                size_t count) -> jsi::Object
            {
                return FOCV_Object::convertToJSI(runtime, arguments);
            });
    }
    else if (propName == "copyObjectFromVector")
    {
        return jsi::Function::createFromHostFunction(
            runtime, jsi::PropNameID::forAscii(runtime, "copyObjectFromVector"), 1,
            [=](jsi::Runtime &runtime, const jsi::Value &thisValue, const jsi::Value *arguments,
                size_t count) -> jsi::Object
            {
                return FOCV_Object::copyObjectFromVector(runtime, arguments);
            });
    }
    else if (propName == "invoke")
    {
        return jsi::Function::createFromHostFunction(
            runtime, jsi::PropNameID::forAscii(runtime, "invoke"), 1,
            [=](jsi::Runtime &runtime, const jsi::Value &thisValue, const jsi::Value *arguments,
                size_t count) -> jsi::Object
            {
                return FOCV_Function::invoke(runtime, arguments);
            });
    }
    else if (propName == "clearBuffers")
    {
        return jsi::Function::createFromHostFunction(
            runtime, jsi::PropNameID::forAscii(runtime, "clearBuffers"), 1,
            [=](jsi::Runtime &runtime, const jsi::Value &thisValue, const jsi::Value *arguments,
                size_t count) -> jsi::Value
            {
                FOCV_Storage::clear();
                return true;
            });
    }
    else if (propName == "getMatData")
    {
        return jsi::Function::createFromHostFunction(
            runtime, jsi::PropNameID::forAscii(runtime, "getMatData"), 1,
            [=](jsi::Runtime &runtime, const jsi::Value &thisValue, const jsi::Value *arguments,
                size_t count) -> jsi::Object
            {
                jsi::Object value(runtime);

                std::string id = FOCV_JsiObject::id_from_wrap(runtime, arguments[0]);
                auto mat = *FOCV_Storage::get<cv::Mat>(id);
                mat.convertTo(mat, CV_8U);

                // Create a TypedArray to hold the Mat data
                size_t dataLength = mat.total() * mat.elemSize();
                std::vector<uint8_t> matData(mat.data, mat.data + dataLength);

                value.setProperty(runtime, "data", TypedArray<TypedArrayKind::Uint8Array>(runtime, matData));
                value.setProperty(runtime, "size", jsi::Value(mat.size));
                value.setProperty(runtime, "cols", jsi::Value(mat.cols));
                value.setProperty(runtime, "rows", jsi::Value(mat.rows));
                
                return value;
            });
    }
    else if (propName == "getMatRoi")
    {
        return jsi::Function::createFromHostFunction(
            runtime, jsi::PropNameID::forAscii(runtime, "getMatData"), 1,
            [=](jsi::Runtime &runtime, const jsi::Value &thisValue, const jsi::Value *arguments,
                size_t count) -> jsi::Object
            {
                // arg: mat, roiRect
                jsi::Object value(runtime);
                std::string matId = FOCV_JsiObject::id_from_wrap(runtime, arguments[0]);
                std::string rectId = FOCV_JsiObject::id_from_wrap(runtime, arguments[1]);

                auto mat = *FOCV_Storage::get<cv::Mat>(matId);
                auto roiRect = *FOCV_Storage::get<cv::Rect>(rectId);

                auto image_rect = cv::Rect({}, mat.size());

                auto intersection = image_rect & roiRect;

                // Move intersection to the result coordinate space
                auto inter_roi = intersection - roiRect.tl();

                // Create black image and copy intersection
                cv::Mat crop = cv::Mat::zeros(roiRect.size(), mat.type());
                mat(intersection).copyTo(crop(inter_roi));

                std::string id = "";
                id = FOCV_Storage::save(crop);
                
                return FOCV_JsiObject::wrap(runtime, "mat", id);
            });
    }

    return jsi::HostObject::get(runtime, propNameId);
}

std::vector<jsi::PropNameID> OpenCVPlugin::getPropertyNames(jsi::Runtime &runtime)
{
    std::vector<jsi::PropNameID> result;

    result.push_back(jsi::PropNameID::forAscii(runtime, "frameBufferToMat"));
    result.push_back(jsi::PropNameID::forAscii(runtime, "bufferToMat"));
    result.push_back(jsi::PropNameID::forAscii(runtime, "base64ToMat"));
    result.push_back(jsi::PropNameID::forAscii(runtime, "matToBuffer"));
    result.push_back(jsi::PropNameID::forAscii(runtime, "createObject"));
    result.push_back(jsi::PropNameID::forAscii(runtime, "toJSValue"));
    result.push_back(jsi::PropNameID::forAscii(runtime, "copyObjectFromVector"));
    result.push_back(jsi::PropNameID::forAscii(runtime, "invoke"));
    result.push_back(jsi::PropNameID::forAscii(runtime, "clearBuffers"));
    result.push_back(jsi::PropNameID::forAscii(runtime, "getMatData"));
    result.push_back(jsi::PropNameID::forAscii(runtime, "getMatRoi"));

    return result;
}
