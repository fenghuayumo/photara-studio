#include "sfm/checkpoint.hpp"
#include "sfm/reconstruct.hpp"
#include "sfm/asfm.hpp"
#include "core/logging.hpp"
#include <chrono>
#include <iostream>

int main(int argc,char** argv){try{
    using namespace photara::sfm;
    photara::core::Logger::instance().configure({});
    if(argc!=4)throw std::invalid_argument("Usage: photara_sfm_positioning_benchmark tracks-HEX.bin cpu|cuda NEW_OUTPUT.asfm");
    const std::filesystem::path input(argv[1]),output(argv[3]);
    if(std::filesystem::exists(output))throw std::invalid_argument("Output already exists");
    auto stem=input.stem().string();auto key=std::stoull(stem.substr(stem.find('-')+1),nullptr,16);
    CheckpointOptions options;options.directory=input.parent_path();options.write=false;Scene scene;
    if(!CheckpointStore(options).load_scene(CheckpointStage::tracks,key,scene))throw std::runtime_error("Cannot load tracks checkpoint");
    GlobalPositioningOptions positioning;positioning.prefer_cuda=std::string(argv[2])=="cuda";
    if(!positioning.prefer_cuda&&std::string(argv[2])!="cpu")throw std::invalid_argument("Unknown backend");
    const auto start=std::chrono::steady_clock::now();
    auto result=run_global_mapping(scene,{},positioning,{});
    std::cout<<"mapping_seconds="<<std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()
        <<" registered="<<scene.registered_count()<<" landmarks="<<result.landmarks<<'\n';
    save_asfm(scene,output);return scene.registered_count()?0:2;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
