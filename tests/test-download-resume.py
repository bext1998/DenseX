#!/usr/bin/env python3
"""DenseX：模型下載器續傳與完整性的 loopback 測試（issue #78）。

mock 是 127.0.0.1 上的標準 http.server；C++ 驅動程式只呼叫
common_download_file_single(url, dest)。每個情境都從乾淨目錄開始，檢查完成檔、
殘留的暫存檔，以及請求標頭。

情境：
  1. 版本 A -> B          暫存檔屬於 A，伺服器已改成 B
  2. 錯誤 Content-Range   206 的起點不是我們要的位移
  3. 長度不符             body 比宣告的大小短
  4. 摘要不符             body 完整，但 SHA-256 與 X-Linked-ETag 不同
  5. 正常續傳             If-Range + 206，完成檔必須逐位元組相同
  6. If-Range 被忽略      伺服器對續傳回 200 與完整 body
  7. 大檔續傳             超過 1 MiB 雜湊緩衝區
  8. SHA-256 padding      55/56/64/65 位元組邊界
  9. 沒有可信摘要         一般伺服器仍可用強 ETag 續傳
 10. 轉址與 X-Linked-ETag HF 的 302 形狀

用法：python test-download-resume.py <test-download-resume 執行檔路徑>
"""

import hashlib
import http.server
import os
import shutil
import subprocess
import sys
import tempfile
import threading

SIZE = 4096
PARTIAL = 1000
LM = "Wed, 19 Nov 2026 00:00:00 GMT"


def sha256(data):
    return hashlib.sha256(data).hexdigest()


class State:
    def __init__(self, content, etag, linked_sha256=None, bad_content_range=False,
                 short_body=False, ignore_range=False, redirect=False):
        self.content = content
        self.etag = etag
        self.last_modified = LM
        self.linked_sha256 = linked_sha256
        self.bad_content_range = bad_content_range
        self.short_body = short_body
        self.ignore_range = ignore_range
        self.redirect = redirect


class MockServer(http.server.ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, state):
        super().__init__(("127.0.0.1", 0), MockHandler)
        self.state = state
        self.requests = []


class MockHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def _send_headers(self, length, status=200, extra=None):
        self.send_response(status)
        self.send_header("Content-Length", str(length))
        self.send_header("ETag", self.server.state.etag)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Last-Modified", self.server.state.last_modified)
        if self.server.state.linked_sha256 is not None:
            self.send_header("X-Linked-ETag", '"%s"' % self.server.state.linked_sha256)
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()

    def _send_body(self, body):
        if self.server.state.short_body:
            self.wfile.write(body[: max(1, len(body) // 2)])
            self.wfile.flush()
            self.close_connection = True
            return
        self.wfile.write(body)

    def _maybe_redirect(self):
        state = self.server.state
        if not state.redirect or self.path != "/model.gguf":
            return False
        self.send_response(302)
        self.send_header("Location", "/blob")
        self.send_header("Content-Length", "0")
        if state.linked_sha256 is not None:
            self.send_header("X-Linked-ETag", '"%s"' % state.linked_sha256)
        self.end_headers()
        return True

    def do_HEAD(self):
        self.server.requests.append({
            "method": "HEAD",
            "range": self.headers.get("Range"),
            "if_range": self.headers.get("If-Range"),
        })
        if self._maybe_redirect():
            return
        self._send_headers(len(self.server.state.content))

    def do_GET(self):
        state = self.server.state
        req = {
            "method": "GET",
            "range": self.headers.get("Range"),
            "if_range": self.headers.get("If-Range"),
        }
        self.server.requests.append(req)
        if self._maybe_redirect():
            return

        content = state.content
        total = len(content)
        use_range = req["range"] is not None and not state.ignore_range
        if use_range and req["if_range"] is not None and req["if_range"] != state.etag:
            use_range = False

        try:
            if use_range:
                first, last = req["range"].split("=", 1)[1].split("-", 1)
                start = int(first)
                end = int(last) if last else total - 1
                end = min(end, total - 1)
                if start >= total:
                    self._send_headers(0, status=416)
                    return
                body = content[start:end + 1]
                cr_start, cr_end = start, end
                if state.bad_content_range:
                    cr_start = start + 1 if start + 1 <= end else 0
                self._send_headers(len(body), status=206,
                                   extra={"Content-Range": "bytes %d-%d/%d" % (cr_start, cr_end, total)})
                self._send_body(body)
            else:
                self._send_headers(total)
                self._send_body(content)
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True


def partial_len(content):
    return min(PARTIAL, max(1, len(content) // 2))


def write_partial(dest, content, etag, digest, size, url):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest + ".downloadInProgress", "wb") as f:
        f.write(content[:partial_len(content)])
    with open(dest + ".downloadInProgress.meta", "w", encoding="utf-8") as f:
        f.write("version=1\n")
        f.write("url=%s\n" % url)
        f.write("etag=%s\n" % etag)
        f.write("last-modified=%s\n" % LM)
        f.write("size=%d\n" % size)
        f.write("sha256=%s\n" % digest)


class Case:
    def __init__(self, name, state, prepare, check):
        self.name = name
        self.state = state
        self.prepare = prepare
        self.check = check


def run_case(exe, case):
    server = MockServer(case.state)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = "http://127.0.0.1:%d/model.gguf" % server.server_address[1]
        workdir = tempfile.mkdtemp(prefix="densex-dl-")
        dest = os.path.join(workdir, "model.gguf")
        if case.prepare:
            case.prepare(dest, url)
        proc = subprocess.run([exe, url, dest], capture_output=True, text=True, timeout=120)
        case.check(dest, proc, server.requests)
        return True, ""
    except AssertionError as exc:
        detail = str(exc)
        if "proc" in locals():
            detail += "\n--- stdout ---\n%s\n--- stderr ---\n%s" % (proc.stdout[-4000:], proc.stderr[-8000:])
        return False, detail
    finally:
        server.shutdown()
        server.server_close()
        if "workdir" in locals():
            shutil.rmtree(workdir, ignore_errors=True)


def read(path):
    with open(path, "rb") as f:
        return f.read()


def size_or_zero(path):
    return os.path.getsize(path) if os.path.exists(path) else 0


def partial_prepare(content, etag, digest=None):
    def prepare(dest, url):
        write_partial(dest, content, etag, sha256(content) if digest is None else digest, len(content), url)
    return prepare


def main():
    if len(sys.argv) != 2:
        print("usage: test-download-resume.py <path-to-driver>")
        return 2
    exe = sys.argv[1]
    if not os.path.isfile(exe):
        print("driver not found: %s" % exe)
        return 2

    b = b"B" * SIZE
    a_etag = '"etag-a"'
    b_etag = '"etag-b"'
    b_sha = sha256(b)

    cases = []

    def success_case(name, state, content, etag, check_requests=None, digest=None):
        def check(dest, proc, requests):
            assert proc.returncode == 0, "%s: expected success, got %d" % (name, proc.returncode)
            assert os.path.exists(dest), "%s: final file missing" % name
            got = read(dest)
            assert got == content, "%s: final file is wrong or mixes versions (%d bytes)" % (name, len(got))
            assert not os.path.exists(dest + ".downloadInProgress"), "%s: temp file left behind" % name
            if check_requests is not None:
                check_requests(requests, content)
        return Case(name, state, partial_prepare(content, etag, digest), check)

    # 1. 版本 A -> B：暫存檔是 A，伺服器現在供應 B。
    def check_ab(requests, content):
        assert not any(r.get("if_range") for r in requests if r["method"] == "GET"), \
            "A->B: a request carried If-Range even though the version changed"

    cases.append(success_case("version A -> B", State(b, b_etag, b_sha), b, a_etag, check_ab))

    # 2. 錯誤的 Content-Range：206 宣告的起點位移不對。
    def check_bad_cr(dest, proc, requests):
        assert proc.returncode != 0, "bad Content-Range: download must fail"
        assert not os.path.exists(dest), "bad Content-Range: temp file was renamed to the final file"
        assert size_or_zero(dest + ".downloadInProgress") == partial_len(b), \
            "bad Content-Range: partial file was modified before the header was accepted"

    cases.append(Case("bad Content-Range", State(b, b_etag, b_sha, bad_content_range=True),
                      partial_prepare(b, b_etag), check_bad_cr))

    # 3. 長度不符：body 比 Content-Length／Content-Range 宣告的短。
    def check_short(dest, proc, requests):
        assert proc.returncode != 0, "length mismatch: download must fail"
        assert not os.path.exists(dest), "length mismatch: incomplete file was renamed to the final file"

    cases.append(Case("length mismatch", State(b, b_etag, b_sha, short_body=True),
                      partial_prepare(b, b_etag), check_short))

    # 4. 摘要不符：body 完整，但 X-Linked-ETag 是另一個 SHA-256。
    #    既有的有效舊檔必須在更新失敗後原封不動。
    old = b"O" * SIZE

    def prepare_digest(dest, url):
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "wb") as f:
            f.write(old)

    def check_digest(dest, proc, requests):
        assert proc.returncode != 0, "digest mismatch: download must fail"
        assert os.path.exists(dest), "digest mismatch: the previous valid file was deleted"
        assert read(dest) == old, "digest mismatch: the previous valid file was replaced"
        assert not os.path.exists(dest + ".downloadInProgress") or \
            size_or_zero(dest + ".downloadInProgress") == 0, \
            "digest mismatch: corrupt data was left as a resumable partial file"

    cases.append(Case("digest mismatch", State(b, b_etag, sha256(b"other content")),
                      prepare_digest, check_digest))

    # 5. 正常續傳：If-Range + 206，最後必須是遠端檔案的完整內容。
    def resume_check(name, etag):
        def check(requests, content):
            gets = [r for r in requests if r["method"] == "GET"]
            assert gets, "%s: no GET request" % name
            assert gets[0]["range"] == "bytes=%d-" % partial_len(content), \
                "%s: Range was %r, expected bytes=%d-" % (name, gets[0]["range"], partial_len(content))
            assert gets[0]["if_range"] == etag, \
                "%s: If-Range was %r, expected %r" % (name, gets[0]["if_range"], etag)
        return check

    cases.append(success_case("normal resume", State(b, b_etag, b_sha), b, b_etag, resume_check("resume", b_etag)))

    # 6. If-Range 被忽略：伺服器對帶 Range 的請求回 200 與整個檔案。
    #    下載器必須丟掉舊前綴，不能接上去。
    def check_restart(requests, content):
        gets = [r for r in requests if r["method"] == "GET"]
        assert gets and gets[0]["range"] == "bytes=%d-" % partial_len(content), \
            "ignored If-Range: the resume request did not carry the expected Range"

    cases.append(success_case("ignored If-Range", State(b, b_etag, b_sha, ignore_range=True),
                              b, b_etag, check_restart))

    # 7. 超過 1 MiB 雜湊緩衝區的大檔，從中段續傳。
    big = (bytes(range(256)) * 10000)[:2500137]
    big_etag = '"etag-big"'
    cases.append(success_case("large resume", State(big, big_etag, sha256(big)), big, big_etag,
                              resume_check("large resume", big_etag)))

    # 8. SHA-256 padding 邊界（55/56/64/65 位元組），帶短續傳。
    for n in (55, 56, 64, 65):
        content = bytes([(i * 37 + n) & 0xFF for i in range(n)])
        etag = '"etag-pad-%d"' % n
        cases.append(success_case("sha256 padding %d" % n, State(content, etag, sha256(content)),
                                  content, etag, resume_check("padding %d" % n, etag)))

    # 9. 沒有 X-Linked-ETag 的一般伺服器：仍可用強 ETag 續傳，
    #    但只能驗證 Content-Range 與大小。
    cases.append(success_case("resume without digest", State(b, b_etag, None), b, b_etag,
                              resume_check("resume without digest", b_etag), digest=""))

    # 10. HF 的形狀：302 轉到類 CDN 路徑，X-Linked-ETag 放在轉址回應上。
    def check_redirect_full(dest, proc, requests):
        assert proc.returncode == 0, "redirect: expected success, got %d" % proc.returncode
        assert read(dest) == b, "redirect: final file is wrong"
        assert not os.path.exists(dest + ".downloadInProgress"), "redirect: temp file left behind"

    cases.append(Case("redirect full download", State(b, b_etag, b_sha, redirect=True), None,
                      check_redirect_full))

    def check_redirect_resume(requests, content):
        assert any(r["method"] == "GET" for r in requests), "redirect resume: no GET request"

    cases.append(success_case("redirect resume", State(b, b_etag, b_sha, redirect=True), b, b_etag,
                              check_redirect_resume))

    failed = 0
    for case in cases:
        ok, detail = run_case(exe, case)
        print("%s  %s" % ("PASS" if ok else "FAIL", case.name))
        if not ok:
            failed += 1
            print(detail)

    print("download resume tests: %d passed, %d failed" % (len(cases) - failed, failed))
    print("DOWNLOAD_RESUME_TESTS_PASS" if failed == 0 else "DOWNLOAD_RESUME_TESTS_FAIL")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
