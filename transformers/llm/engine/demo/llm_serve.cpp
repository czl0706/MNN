#include "llm/llm.hpp"
// #include <audio/audio.hpp>
#include <MNN/AutoTime.hpp>
#include <MNN/expr/ExecutorScope.hpp>
#include "httplib.h"
#include <iostream>
#include <memory>
#include <sstream>

using namespace MNN::Transformer;

int main(int argc, const char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " config.json" << std::endl;
        return 1;
    }

    std::string config_path = argv[1];
    std::cout << "config path is " << config_path << std::endl;

    // 初始化 MNN 執行器
    MNN::BackendConfig backendConfig;
    auto executor = MNN::Express::Executor::newExecutor(MNN_FORWARD_CPU, backendConfig, 1);
    MNN::Express::ExecutorScope s(executor);
    
    // 加載模型
    std::unique_ptr<Llm> llm(Llm::createLLM(config_path));
    llm->set_config("{\"tmp_path\":\"tmp\"}");
    {
        AUTOTIME;
        llm->load();
    }

    // // 設定音訊回調函數
    // std::vector<float> waveform;
    // llm->setWavformCallback([&](const float* ptr, size_t size, bool last_chunk) {
    //     waveform.reserve(waveform.size() + size);
    //     waveform.insert(waveform.end(), ptr, ptr + size);
    //     if (last_chunk) {
    //         auto waveform_var = MNN::Express::_Const(waveform.data(), {(int)waveform.size()}, MNN::Express::NCHW, halide_type_of<float>());
    //         MNN::AUDIO::save("output.wav", waveform_var, 24000);
    //         waveform.clear();
    //     }
    //     return true;
    // });
    
    // HTTP 伺服器設定
    httplib::Server svr;

    // 健康檢查
    svr.Get("/health", [](const httplib::Request& req, httplib::Response& res) {
        res.set_content("OK", "text/plain");
    });

    // Streaming 服務
    svr.Post("/stream", [&llm](const httplib::Request& req, httplib::Response& res) {
        std::string prompt = req.body;
        std::cout << "Received prompt: " << prompt << std::endl;

        llm->reset();
        auto context = llm->getContext();

        // 時間 & 統計變數
        int prompt_len = 0, decode_len = 0;
        long long prefill_time = 0, decode_time = 0, sample_time = 0;

        res.set_chunked_content_provider("text/plain",
            [&, prompt](size_t offset, httplib::DataSink& sink) {
                // 用 DataSink 代替 std::ostream，實現即時輸出
                struct SinkStreamBuf : std::streambuf {
                    httplib::DataSink& sink;
                    SinkStreamBuf(httplib::DataSink& s) : sink(s) {}
                    int overflow(int c) override {
                        if (c != EOF) {
                            char z = c;
                            sink.write(&z, 1);
                            // sink.flush();
                        }
                        return c;
                    }
                } sbuf(sink);
                std::ostream os(&sbuf);

                // 先呼叫 response 生成第一步
                llm->response(prompt, &os, nullptr, 0);

                // 然後開始逐步生成
                while (!llm->stoped()) {
                    llm->generate(1);  // 每次生成1 token
                }


                // // 統計資訊收集
                // prompt_len += context->prompt_len;
                // decode_len += context->gen_seq_len;
                // prefill_time += context->prefill_us;
                // decode_time += context->decode_us;
                // sample_time += context->sample_us;

                // // llm->generateWavform();

                // // 效能統計結果（直接串到輸出）
                // float prefill_s = prefill_time / 1e6;
                // float decode_s = decode_time / 1e6;
                // float sample_s = sample_time / 1e6;

                // std::ostringstream stats;
                // stats << "\n#################################\n";
                // stats << "prompt tokens num = " << prompt_len << "\n";
                // stats << "decode tokens num = " << decode_len << "\n";
                // stats << "prefill time = " << prefill_s << " s\n";
                // stats << " decode time = " << decode_s << " s\n";
                // stats << " sample time = " << sample_s << " s\n";
                // stats << "prefill speed = " << (prefill_s > 0 ? prompt_len / prefill_s : 0) << " tok/s\n";
                // stats << " decode speed = " << (decode_s > 0 ? decode_len / decode_s : 0) << " tok/s\n";
                // stats << "#################################\n";
    
                // std::string stats_str = stats.str();
                // std::cout << "即將輸出統計資訊..." << std::endl;
                // sink.write(stats_str.c_str(), stats_str.size());
    
                

                sink.done();  // 最後通知結束
                return true;
            }
        );
    });

    // 主要的服務 endpoint
    svr.Post("/generate", [&llm](const httplib::Request& req, httplib::Response& res) {
        // 取得使用者傳來的 prompt
        std::string prompt = req.body;
        std::cout << "Received prompt: " << prompt << std::endl;

        // 清除上次的狀態（若需要）
        llm->reset();
        auto context = llm->getContext();

        // 時間 & 統計變數
        int prompt_len = 0, decode_len = 0;
        long long prefill_time = 0, decode_time = 0, sample_time = 0;    

        // 呼叫模型取得回應
        std::ostringstream oss;
        llm->response(prompt, &oss);
        // llm->generateWavform();

        // std::string output = oss.str();
        // std::cout << "Model output: " << output << std::endl;


        // 統計資訊收集
        prompt_len += context->prompt_len;
        decode_len += context->gen_seq_len;
        prefill_time += context->prefill_us;
        decode_time += context->decode_us;
        sample_time += context->sample_us;

        // llm->generateWavform();

        // 統計結果串到輸出中
        float prefill_s = prefill_time / 1e6;
        float decode_s = decode_time / 1e6;
        float sample_s = sample_time / 1e6;

        oss << "\n#################################\n";
        oss << "prompt tokens num = " << prompt_len << "\n";
        oss << "decode tokens num = " << decode_len << "\n";
        oss << "prefill time = " << prefill_s << " s\n";
        oss << " decode time = " << decode_s << " s\n";
        oss << " sample time = " << sample_s << " s\n";
        oss << "prefill speed = " << (prefill_s > 0 ? prompt_len / prefill_s : 0) << " tok/s\n";
        oss << " decode speed = " << (decode_s > 0 ? decode_len / decode_s : 0) << " tok/s\n";
        oss << "#################################\n";

        std::string output = oss.str();
        std::cout << "Final output:\n" << output << std::endl;


        // 回傳結果
        res.set_content(output, "text/plain");
    });

    // 監聽 port（可依需要修改 port）
    int port = 8080;
    std::cout << "Server listening on port " << port << "..." << std::endl;
    svr.listen("0.0.0.0", port);

    return 0;
}
