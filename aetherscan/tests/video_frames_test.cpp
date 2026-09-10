#include "io/image.hpp"
#include "io/video_frames.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int failures = 0;

void expect(const bool condition, const char* message) {
    if (condition) return;
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
}

aetherscan::io::RgbImage make_image(
    const std::uint32_t width, const std::uint32_t height, const std::uint8_t value) {
    aetherscan::io::RgbImage image;
    image.width = width;
    image.height = height;
    image.pixels.assign(static_cast<std::size_t>(width) * height * 3, value);
    return image;
}

aetherscan::io::RgbImage make_checkerboard(
    const std::uint32_t width, const std::uint32_t height) {
    auto image = make_image(width, height, 0);
    for (std::uint32_t y = 0; y < height; ++y) {
        for (std::uint32_t x = 0; x < width; ++x) {
            const std::uint8_t value =
                (((x / 4) + (y / 4)) % 2 == 0) ? 255 : 0;
            const std::size_t i = (static_cast<std::size_t>(y) * width + x) * 3;
            image.pixels[i] = value;
            image.pixels[i + 1] = value;
            image.pixels[i + 2] = value;
        }
    }
    return image;
}

}  // namespace

int main() {
    expect(aetherscan::io::is_video_path("clip.mp4"), "mp4 is video");
    expect(aetherscan::io::is_video_path("CLIP.MOV"), "MOV is video");
    expect(aetherscan::io::is_video_path("dual.insv"), "insv is video");
    expect(!aetherscan::io::is_video_path("photo.jpg"), "jpg is not video");
    expect(!aetherscan::io::is_video_path("images"), "folder is not video");

    const auto frames = aetherscan::io::default_video_frames_dir("D:/scan/clip.mp4");
    expect(
        frames.filename() == "images" && frames.parent_path().filename() == "clip",
        "default frames dir is <stem>/images");
    {
        // UTF-8 D:/<U+626B U+63CF>/clip.mp4, not decoded as ACP.
        const char utf8[] = "D:/\xE6\x89\xAB\xE6\x8F\x8F/clip.mp4";
        const auto* begin = reinterpret_cast<const char8_t*>(utf8);
        const std::filesystem::path video(
            std::u8string(begin, begin + sizeof(utf8) - 1));
        const auto chinese = aetherscan::io::default_video_frames_dir(video);
        expect(
            chinese.filename() == "images" &&
                chinese.parent_path().filename() == "clip" &&
                chinese.parent_path().parent_path().filename().u8string() ==
                    std::u8string(u8"\u626B\u63CF"),
            "default frames dir keeps a non-ASCII parent");
    }

    const auto sharp = make_checkerboard(64, 64);
    const auto blur = make_image(64, 64, 128);
    const double sharp_score = aetherscan::io::laplacian_sharpness(sharp);
    const double blur_score = aetherscan::io::laplacian_sharpness(blur);
    expect(sharp_score > blur_score * 5.0, "checkerboard is sharper than flat");
    expect(sharp_score > 1.0, "checkerboard score is positive");

    const auto dir = std::filesystem::temp_directory_path() / "aetherscan_video_frames_test";
    const auto cand = dir / "cand";
    const auto out = dir / "kept";
    std::error_code error;
    std::filesystem::remove_all(dir, error);
    std::filesystem::create_directories(cand, error);

    for (int i = 0; i < 4; ++i) {
        char name[32];
        std::snprintf(name, sizeof(name), "c_%02d.png", i);
        aetherscan::io::save_rgb_png(
            i % 2 == 0 ? blur : sharp, cand / name);
    }
    const int kept = aetherscan::io::select_sharpest_frames(cand, out, 2, 0);
    expect(kept == 2, "keep one of each pair");
    expect(
        std::filesystem::exists(out / "00000.png") &&
            std::filesystem::exists(out / "00001.png"),
        "selected frames are numbered");
    const double kept0 = aetherscan::io::laplacian_sharpness(out / "00000.png");
    const double kept1 = aetherscan::io::laplacian_sharpness(out / "00001.png");
    expect(kept0 > blur_score * 5.0 && kept1 > blur_score * 5.0,
           "kept frames are the sharp ones");

    {
        const auto dummy = dir / "dummy.mp4";
        const auto occupied = dir / "occupied";
        std::filesystem::create_directories(occupied, error);
        {
            std::ofstream(dummy) << "not a video";
        }
        aetherscan::io::save_rgb_png(sharp, occupied / "photo.png");
        aetherscan::io::VideoExtractOptions occupied_job;
        occupied_job.video = dummy;
        occupied_job.output_dir = occupied;
        occupied_job.sharp_window = 1;
        bool threw = false;
        try {
            aetherscan::io::extract_video_frames(occupied_job);
        } catch (const std::exception&) {
            threw = true;
        }
        expect(threw, "occupied folder is refused");
        expect(
            std::filesystem::exists(occupied / "photo.png"),
            "user photo is not deleted");
        expect(
            !aetherscan::io::has_video_extract_manifest(occupied),
            "occupied folder gets no extract manifest");
        expect(
            !aetherscan::io::has_matching_video_extract(occupied_job),
            "occupied folder is not a matching extract");
    }

    if (aetherscan::io::ffmpeg_available()) {
        const auto video = dir / "testsrc.mp4";
        const auto extracted = dir / "extracted";
        const std::string synth =
            "ffmpeg -nostdin -y -hide_banner -loglevel error -f lavfi -i "
            "testsrc=duration=1:size=64x64:rate=10 -pix_fmt yuv420p \"" +
            video.string() + "\"";
        if (std::system(synth.c_str()) == 0 &&
            std::filesystem::exists(video, error)) {
            try {
                aetherscan::io::VideoExtractOptions options;
                options.video = video;
                options.output_dir = extracted;
                options.fps = 5.0F;
                options.sharp_window = 2;
                options.max_frames = 8;
                options.quality = 90;
                const auto result = aetherscan::io::extract_video_frames(options);
                expect(result.frames_written >= 2, "ffmpeg extract writes frames");
                expect(
                    std::filesystem::exists(extracted / "00000.jpg"),
                    "extracted frame 00000.jpg");
                const auto again = aetherscan::io::extract_video_frames(options);
                expect(again.reused, "second extract reuses matching frames");
            } catch (const std::exception& failure) {
                std::cerr << "FAIL: ffmpeg extract: " << failure.what() << '\n';
                ++failures;
            }
        } else {
            std::cout << "ffmpeg could not write a test clip; skipping decode\n";
        }
    } else {
        std::cout << "ffmpeg not on PATH; skipping decode test\n";
    }

    std::filesystem::remove_all(dir, error);
    if (failures > 0) {
        std::cerr << failures << " failures\n";
        return 1;
    }
    std::cout << "video_frames_test ok\n";
    return 0;
}
