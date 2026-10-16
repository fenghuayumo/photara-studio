#include "model_download.hpp"
#include "moge_model.hpp"
#include "core/sha256.hpp"
#if defined(PHOTARA_HAS_INFERENCE)
#include "inference/models/moge.hpp"
#endif
#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <thread>

namespace {
void check(bool value,const char* message) {if(!value)throw std::runtime_error(message);}
void write(const std::filesystem::path& path,const std::string& text) {
    std::ofstream out(path,std::ios::binary);out<<text;
    check(bool(out),"Cannot write test fixture");
}
void wait(editor::ModelDownload& download) {
    const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(15);
    while(download.state()==editor::ModelDownload::State::running) {
        if(std::chrono::steady_clock::now()>deadline) {download.cancel();throw std::runtime_error("Download test timed out");}
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}
std::string file_url(const std::filesystem::path& path) {
    const auto u8=std::filesystem::absolute(path).generic_u8string();
    std::string result="file://";
#if defined(_WIN32)
    result+='/';
#endif
    constexpr char digits[]="0123456789ABCDEF";
    for(unsigned char c:u8) {
        if((c>='a'&&c<='z')||(c>='A'&&c<='Z')||(c>='0'&&c<='9')||c=='/'||c==':'||c=='.'||c=='-'||c=='_')result+=c;
        else {result+='%';result+=digits[c>>4];result+=digits[c&15];}
    }
    return result;
}
}
int main(int argc,char** argv) {
    try {
        if(argc==3&&std::string(argv[1])=="--remote") {
            editor::ModelDownloadSpec spec;
            spec.url=editor::k_moge_model_url;spec.destination=argv[2];
            spec.expected_bytes=editor::k_moge_model_bytes;spec.sha256=editor::k_moge_model_sha256;
#if defined(PHOTARA_HAS_INFERENCE)
            spec.validate=[](const auto& path) {(void)photara::inference::inspect_moge_model(path);};
#endif
            editor::ModelDownload download;download.start(spec);
            auto next=std::chrono::steady_clock::now();
            while(download.state()==editor::ModelDownload::State::running) {
                if(std::chrono::steady_clock::now()>=next) {
                    std::cout<<download.progress()<<" "<<download.status()<<'\n'<<std::flush;
                    next=std::chrono::steady_clock::now()+std::chrono::seconds(5);
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
            check(download.state()==editor::ModelDownload::State::done,download.status().c_str());
            check(editor::verified_download_ready(spec),"Missing remote verification receipt");
            std::cout<<"Release GGUF downloaded and SHA-256/format verified\n";return 0;
        }
        check(argc==1,"Usage: model_download_test [--remote OUTPUT]");
        const auto root=std::filesystem::temp_directory_path()/
            ("photara-model-download-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directories(root);
        const auto fixture=root/"source model.bin";
        write(fixture,"");
        check(photara::core::sha256_file(fixture)=="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","Empty SHA-256 vector");
        write(fixture,std::string(1000000,'a'));
        check(photara::core::sha256_file(fixture)=="cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0","Streaming SHA-256 vector");
        write(fixture,"abc");
        editor::ModelDownloadSpec spec;
        spec.url=file_url(fixture);spec.destination=root/"cached model.bin";spec.expected_bytes=3;
        spec.sha256="ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
        auto partial=spec.destination;partial+=".part";
        write(partial,"ab"); // Resumable transfer, including paths with spaces.
        editor::ModelDownload download;download.start(spec);wait(download);
        check(download.state()==editor::ModelDownload::State::done,download.status().c_str());
        check(editor::verified_download_ready(spec),"Verified receipt not reusable");
        check(!std::filesystem::exists(partial),"Partial file was published incorrectly");
        auto offline=spec;offline.url=file_url(root/"does-not-exist");
        download.start(offline);wait(download);
        check(download.state()==editor::ModelDownload::State::done,"Existing valid model should work offline");
        write(spec.destination,"bad");
        std::filesystem::last_write_time(spec.destination,std::filesystem::file_time_type::clock::now()+std::chrono::seconds(2));
        check(!editor::verified_download_ready(spec),"Modified model kept a valid receipt");
        auto mismatch=spec;mismatch.sha256=std::string(64,'0');
        download.start(mismatch);wait(download);
        check(download.state()==editor::ModelDownload::State::failed,"SHA mismatch accepted");
        check(photara::core::sha256_file(spec.destination)!=spec.sha256,"Failed download overwrote cached file");
        check(!std::filesystem::exists(partial),"Bad checksum partial was retained");
        download.start(spec);wait(download);
        check(download.state()==editor::ModelDownload::State::done,"Retry did not repair cache");
        auto missing=spec;missing.destination=root/"missing.bin";missing.url=file_url(root/"missing source");
        download.start(missing);wait(download);
        check(download.state()==editor::ModelDownload::State::failed,"Transport failure accepted");
        auto wrong_size=spec;wrong_size.destination=root/"wrong size.bin";wrong_size.expected_bytes=4;
        download.start(wrong_size);wait(download);
        check(download.state()==editor::ModelDownload::State::failed,"Wrong file size accepted");
        auto cancelled=spec;cancelled.destination=root/"cancelled.bin";
        download.start(cancelled);download.cancel();wait(download);
        check(download.state()==editor::ModelDownload::State::cancelled,"Cancellation failed");
        check(!std::filesystem::exists(cancelled.destination),"Cancelled model was published");
        check(std::filesystem::equivalent(root.parent_path(),std::filesystem::temp_directory_path()),"Unexpected fixture cleanup root");
        for (const auto& entry:std::filesystem::directory_iterator(root)) {
            check(entry.is_regular_file(),"Unexpected directory in fixture cleanup");
            std::filesystem::remove(entry.path());
        }
        std::filesystem::remove(root);
        std::cout<<"Model download, resume, verification, cache, retry and cancellation tests passed\n";
        return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
