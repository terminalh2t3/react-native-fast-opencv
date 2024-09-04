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
  else if (propName == "base64ToMat") {
      return jsi::Function::createFromHostFunction(
          runtime, jsi::PropNameID::forAscii(runtime, "frameBufferToMat"), 1,
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

    return jsi::HostObject::get(runtime, propNameId);
}

std::vector<jsi::PropNameID> OpenCVPlugin::getPropertyNames(jsi::Runtime &runtime)
{
    std::vector<jsi::PropNameID> result;

    result.push_back(jsi::PropNameID::forAscii(runtime, "frameBufferToMat"));
    result.push_back(jsi::PropNameID::forAscii(runtime, "base64ToMat"));
    result.push_back(jsi::PropNameID::forAscii(runtime, "matToBuffer"));
    result.push_back(jsi::PropNameID::forAscii(runtime, "createObject"));
    result.push_back(jsi::PropNameID::forAscii(runtime, "toJSValue"));
    result.push_back(jsi::PropNameID::forAscii(runtime, "copyObjectFromVector"));
    result.push_back(jsi::PropNameID::forAscii(runtime, "invoke"));
    result.push_back(jsi::PropNameID::forAscii(runtime, "clearBuffers"));
    result.push_back(jsi::PropNameID::forAscii(runtime, "getMatData"));

    return result;
}
