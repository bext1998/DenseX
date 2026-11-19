// DenseX：下載續傳與完整性的迴路測試驅動程式（issue #78）。
//
// 用法：test-download-resume <url> <dest>
// 只呼叫 common_download_file_single，結束碼 0 表示下載成功；檢查檔案內容與
// 殘留檔案的工作交給 tests/test-download-resume.py，它會啟動 loopback mock
// http.server 並比對結果。
#include "download.h"

#include <cstdio>
#include <exception>

int main(int argc, char ** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s <url> <dest>\n", argv[0]);
        return 2;
    }

    common_download_opts opts;
    int status = -1;
    try {
        status = common_download_file_single(argv[1], argv[2], opts);
    } catch (const std::exception & e) {
        std::fprintf(stderr, "exception: %s\n", e.what());
        return 3;
    }
    std::printf("download status: %d\n", status);

    return (status >= 200 && status < 400) ? 0 : 1;
}
