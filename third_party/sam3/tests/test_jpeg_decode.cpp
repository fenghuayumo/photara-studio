// Compare stb_image JPEG decoding against PIL/libjpeg.
// Usage: ./test_jpeg_decode <image.jpg> <pil_decoded.bin>
#include "sam3.h"
#include "test_utils.h"
#include <cstdio>
#include <cstring>
#include <fstream>

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <image.jpg> <pil_decoded.bin>\n", argv[0]);
        return 1;
    }
    auto img = sam3_load_image(argv[1]);
    if (img.data.empty()) { fprintf(stderr, "Failed to load image\n"); return 1; }
    fprintf(stderr, "stb_image decoded: %dx%d, %d bytes\n",
            img.width, img.height, (int)img.data.size());

    // Load PIL reference
    std::ifstream f(argv[2], std::ios::binary);
    if (!f) { fprintf(stderr, "Failed to load PIL ref\n"); return 1; }
    std::vector<uint8_t> pil_data(img.data.size());
    f.read(reinterpret_cast<char*>(pil_data.data()), pil_data.size());
    fprintf(stderr, "PIL decoded: %d bytes\n", (int)pil_data.size());

    // Compare
    int n_diff = 0, max_diff = 0;
    for (size_t i = 0; i < img.data.size(); ++i) {
        int d = abs((int)img.data[i] - (int)pil_data[i]);
        if (d > 0) n_diff++;
        if (d > max_diff) max_diff = d;
    }
    fprintf(stderr, "\n═══ JPEG Decoder Comparison ═══\n");
    fprintf(stderr, "  max pixel diff: %d\n", max_diff);
    fprintf(stderr, "  pixels differing: %d / %d (%.2f%%)\n",
            n_diff, (int)img.data.size(), 100.0 * n_diff / img.data.size());

    // Check specific pixels
    int y=16, x=517, c=2;
    int idx = (y * img.width + x) * 3 + c;
    fprintf(stderr, "\n  stb pixel at (%d,%d,%d): %d\n", y, x, c, img.data[idx]);
    fprintf(stderr, "  PIL pixel at (%d,%d,%d): %d\n", y, x, c, pil_data[idx]);

    return max_diff > 0 ? 1 : 0;
}
