/*
 * Copyright 2021 Huawei Technologies Co., Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <experimental/filesystem>
#include "InceptionResnetV2Classify.h"
#include <iostream>
#include <vector>
#include "MxBase/Log/Log.h"

namespace fs = std::experimental::filesystem;
namespace {
    const uint32_t CLASS_NUM = 1000;
}
std::vector<double> g_inferCost;

int main(int argc, char* argv[]) {
    if (argc <= 1) {
        LogWarn << "Please input image path, such as './build/inception_resnet_v2 ./image_dir'.";
        return APP_ERR_OK;
    }

    InitParam initParam = {};
    initParam.deviceId = 0;
    initParam.classNum = CLASS_NUM;
    initParam.labelPath = "../data/config/imagenet1000_clsidx_to_labels.names";
    initParam.topk = 5;
    initParam.softmax = true;
    initParam.checkTensor = true;
    initParam.modelPath = "../data/models/inception_resnet_v2.om";
    auto inceptionresentv2 = std::make_shared<InceptionResnetV2Classify>();
    APP_ERROR ret = inceptionresentv2->Init(initParam);
    if (ret != APP_ERR_OK) {
        LogError << "InceptionResentV2Classify init failed, ret=" << ret << ".";
        return ret;
    }

    std::string imgDir = argv[1];
    for (auto & entry : fs::directory_iterator(imgDir)) {
        LogInfo << "read image path " << entry.path();
        ret = inceptionresentv2->Process(entry.path());
        if (ret != APP_ERR_OK) {
            LogError << "InceptionResentV2Classify process failed, ret=" << ret << ".";
            inceptionresentv2->DeInit();
            return ret;
        }
    }
    inceptionresentv2->DeInit();
    double costSum = 0;
    for (unsigned int i = 0; i < g_inferCost.size(); i++) {
        costSum += g_inferCost[i];
    }
    LogInfo << "Infer images sum " << g_inferCost.size() << ", cost total time: " << costSum << " ms.";
    LogInfo << "The throughput: " << g_inferCost.size() * 1000 / costSum << " images/sec.";
    return APP_ERR_OK;
}
