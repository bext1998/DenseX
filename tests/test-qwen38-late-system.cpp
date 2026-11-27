// DenseX（issue #88）：Qwen 3.8 出廠範本對「不是第一則」的 system 訊息 raise_exception，工具鏈在中途插入
// system 就會 500。common/chat.cpp 在載入模型範本時把那段分支換成就地渲染；這個測試驗證替換的正確性與邊界。
// 輸入：只有 vocab 的 GGUF（argv[1]，含出廠 chat template）與修好版對照檔（argv[2]）。
// 輸出：逐字比對與各案例結果；全過時最後一行 QWEN38_LATE_SYSTEM_OK，否則 QWEN38_LATE_SYSTEM_FAILED 並回傳 1。
// 開關：DENSEX_LATE_SYSTEM（預設 1；0 = 不修，維持出廠行為）。測試一開始先清掉這個變數，不受外部 shell 影響。
// 依賴：common/chat.cpp 的 densex_chat_template_fix_late_system()、common_chat_templates_init/apply。

#include <cstdlib>
#include <exception>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "chat.h"
#include "common.h"
#include "json.h"
#include "llama.h"

#include "jinja/caps.h"
#include "jinja/lexer.h"
#include "jinja/parser.h"
#include "jinja/runtime.h"

using json = common_json;

static int failures = 0;

static void check(bool ok, const std::string & what) {
    if (ok) {
        std::cout << "  ok   " << what << "\n";
    } else {
        std::cout << "  FAIL " << what << "\n";
        ++failures;
    }
}

static std::string read_file_or_empty(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        return "";
    }
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

// 把 rendered prompt 的 \r\n 正規化成 \n（Windows 上讀對照檔時差一個 CR 不該算失敗）
static std::string normalize_newlines(std::string s) {
    for (size_t pos = 0; (pos = s.find("\r\n", pos)) != std::string::npos;) {
        s.replace(pos, 2, "\n");
    }
    return s;
}

// 清掉開關，讓「未設定＝預設開」這條檢查跟呼叫端 shell 的環境無關
static void unset_late_system_env() {
#ifdef _WIN32
    _putenv_s("DENSEX_LATE_SYSTEM", "");
#else
    unsetenv("DENSEX_LATE_SYSTEM");
#endif
}

static std::string render(const common_chat_templates * tmpls, common_chat_templates_inputs inputs) {
    inputs.use_jinja = true;
    inputs.now       = std::chrono::system_clock::from_time_t(1700000000); // 固定時間，兩邊才可比
    // add_generation_prompt 由呼叫端決定，這裡不覆寫（08_no_generation_prompt 靠它才是 false）
    return common_chat_templates_apply(tmpls, inputs).prompt.str();
}

// 會丟例外時回傳例外訊息，否則回傳空字串
static std::string render_error(const common_chat_templates * tmpls, common_chat_templates_inputs inputs) {
    try {
        render(tmpls, std::move(inputs));
        return "";
    } catch (const std::exception & e) {
        return std::string(e.what());
    }
}

// 繞過 libcommon，直接用 jinja 引擎：內容可以是 OAI 那樣的 content parts（含 image）
static std::string render_direct_jinja_error(const std::string & tmpl, const std::string & input_json) {
    try {
        json input = json::parse(input_json);
        jinja::lexer lexer;
        auto         lexer_res = lexer.tokenize(tmpl);
        jinja::program ast     = jinja::parse_from_tokens(lexer_res);
        jinja::caps_get(ast);
        jinja::context ctx(tmpl);
        jinja::global_from_json(ctx, input, true);
        jinja::runtime runtime(ctx);
        auto parts = jinja::runtime::gather_string_parts(runtime.execute(ast));
        (void) parts;
        return "";
    } catch (const std::exception & e) {
        return std::string(e.what());
    }
}

// 數出「完整的一個獨立 system 回合」出現幾次：'<|im_start|>system\n' + content + '<|im_end|>\n'。
// 內容要完全相符才算，所以同時驗證了結尾的 <|im_end|>\n；先檢查剩餘長度再切片，避免 out_of_range。
static int count_full_system_turns(const std::string & out, const std::string & content) {
    const std::string turn = "<|im_start|>system\n" + content + "<|im_end|>\n";
    int               n    = 0;
    for (size_t pos = 0; pos + turn.size() <= out.size();) {
        const size_t hit = out.find(turn, pos);
        if (hit == std::string::npos) {
            break;
        }
        ++n;
        pos = hit + turn.size();
    }
    return n;
}

// 有開頭但沒結尾、或內容為空的 system 回合（不該出現）
static bool has_empty_system_turn(const std::string & out) {
    return out.find("<|im_start|>system\n<|im_end|>") != std::string::npos;
}

// 單純數子串出現幾次（不用來比對整回合）
static int count_sub(const std::string & out, const std::string & sub) {
    int n = 0;
    for (size_t pos = 0; (pos = out.find(sub, pos)) != std::string::npos; pos += sub.size()) {
        ++n;
    }
    return n;
}

struct shape_case {
    std::string                   name;
    std::vector<common_chat_msg>  messages;
    std::vector<common_chat_tool> tools;
    bool                          add_generation_prompt;
};

static common_chat_msg msg(const std::string & role, const std::string & content) {
    common_chat_msg m;
    m.role    = role;
    m.content = content;
    return m;
}

static std::vector<common_chat_tool> one_tool() {
    return { { "get_weather", "Get current weather",
               R"({"type":"object","properties":{"city":{"type":"string"}},"required":["city"]})" } };
}

static std::vector<shape_case> normal_shapes() {
    std::vector<shape_case> out;

    // 1. 單輪，沒有 system
    out.push_back({ "01_user_only", { msg("user", "What is 2+2?") }, {}, true });

    // 2. 開頭 system + 單輪
    out.push_back({ "02_system_first", { msg("system", "You are concise."), msg("user", "What is 2+2?") }, {}, true });

    // 3. 多輪
    out.push_back({ "03_multi_turn",
                    { msg("user", "Hello"),
                      msg("assistant", "Hi. How can I help?"),
                      msg("user", "Translate 'cat' to French"),
                      msg("assistant", "chat"),
                      msg("user", "And 'dog'?") },
                    {}, true });

    // 4. 有工具定義
    out.push_back({ "04_tools_defined",
                    { msg("system", "You may use tools."), msg("user", "What is the weather in Taipei?") },
                    one_tool(), true });

    // 5. 工具呼叫 + 工具結果
    {
        common_chat_msg call = msg("assistant", "");
        call.tool_calls      = { { "get_weather", R"({"city":"Taipei"})", "call00001" } };
        common_chat_msg res  = msg("tool", R"(Temperature: 27C)");
        res.tool_name        = "get_weather";
        res.tool_call_id     = "call00001";
        out.push_back({ "05_tool_call_and_result",
                        { msg("system", "You may use tools."), msg("user", "Weather in Taipei?"), call, res },
                        one_tool(), true });
    }

    // 6. 一次工具呼叫、兩個工具結果
    {
        common_chat_msg call = msg("assistant", "");
        call.tool_calls      = { { "get_weather", R"({"city":"Taipei"})", "call00001" },
                                 { "get_weather", R"({"city":"Kaohsiung"})", "call00002" } };
        common_chat_msg r1   = msg("tool", R"(Temperature: 27C)");
        r1.tool_name         = "get_weather";
        r1.tool_call_id      = "call00001";
        common_chat_msg r2   = msg("tool", R"(Temperature: 30C)");
        r2.tool_name         = "get_weather";
        r2.tool_call_id      = "call00002";
        out.push_back({ "06_two_tool_results",
                        { msg("user", "Weather in Taipei and Kaohsiung?"), call, r1, r2 },
                        one_tool(), true });
    }

    // 7. 歷史裡有思考內容（開思考的請求會渲染它）
    out.push_back({ "07_reasoning_history",
                    { msg("system", "You think before answering."),
                      [] {
                          common_chat_msg a = msg("assistant", "The answer is 4.");
                          a.reasoning_content = "2+2 is 4, straightforward.";
                          return a;
                      }(),
                      msg("user", "And 3+3?") },
                    {}, true });

    // 8. 不加 generation prompt（add_generation_prompt=false，渲染結果不該出現結尾的 <|im_start|>assistant）
    out.push_back({ "08_no_generation_prompt",
                    { msg("system", "You are concise."), msg("user", "Hello"), msg("assistant", "Hi") },
                    {}, false });

    // 9. 內容含換行、中日韓、特殊字元與前後空白
    out.push_back({ "09_multiline_and_cjk",
                    { msg("system", "Rule one.\n  Rule two.  "),
                      msg("user", "請用「」包住答案，並回答：<|im_start|> 是不是特殊字元？\n第二行"),
                      msg("assistant", "  「是」  \n\nnext"),
                      msg("user", "再講一次，用日本語") },
                    {}, true });

    return out;
}

static std::map<std::string, std::string> kwargs_off() {
    return { { "enable_thinking", "false" } }; // 正式啟動用的 --chat-template-kwargs
}

static std::map<std::string, std::string> kwargs_think_medium() {
    return { { "reasoning_effort", R"("medium")" } }; // 開思考一律 medium（AGENTS 規則第 8 條）
}

struct kwargs_case {
    std::string                      name;
    std::map<std::string, std::string> kwargs;
    bool                             enable_thinking;
};

static const std::vector<kwargs_case> & kwargs_cases() {
    static const std::vector<kwargs_case> cases = {
        { "thinking_off", kwargs_off(), false },
        { "thinking_medium", kwargs_think_medium(), true },
    };
    return cases;
}

int main(int argc, char ** argv) {
    std::string vocab_path = argc > 1 ? argv[1] : "models/ggml-vocab-qwen38.gguf";
    std::string fixed_path = argc > 2 ? argv[2] : "models/templates/Qwen3.8-27B.jinja";

    unset_late_system_env();

    const std::string fixed_file = normalize_newlines(read_file_or_empty(fixed_path));
    if (fixed_file.empty()) {
        std::cerr << "cannot read " << fixed_path << "\n";
        return 1;
    }

    llama_backend_init();
    auto       mparams = llama_model_default_params();
    mparams.vocab_only = true;
    llama_model * model = llama_model_load_from_file(vocab_path.c_str(), mparams);
    if (!model) {
        std::cerr << "cannot load " << vocab_path << "\n";
        return 1;
    }
    const char * tmpl_raw_c = llama_model_chat_template(model, nullptr);
    if (!tmpl_raw_c || !*tmpl_raw_c) {
        std::cerr << vocab_path << " has no chat template\n";
        return 1;
    }
    const std::string tmpl_raw = tmpl_raw_c;

    std::cout << "\n=== 1. switch and match rules ===\n";
    check(densex_chat_template_late_system_enabled(), "DENSEX_LATE_SYSTEM cleared -> defaults to on");

    common_set_env("DENSEX_LATE_SYSTEM", "0");
    check(!densex_chat_template_late_system_enabled(), "DENSEX_LATE_SYSTEM=0 turns the fix off");
    auto tmpls_off = common_chat_templates_init(model, "");
    check(common_chat_templates_source(tmpls_off.get()) == tmpl_raw, "switch off: model template unchanged");

    unset_late_system_env();
    check(densex_chat_template_late_system_enabled(), "DENSEX_LATE_SYSTEM unset again -> on");
    auto tmpls_on = common_chat_templates_init(model, "");
    check(common_chat_templates_source(tmpls_on.get()) == fixed_file,
          "default on: patched model template == " + fixed_path);

    // 使用者自己指定的範本不改動（這時開關是預設開）
    auto tmpls_user = common_chat_templates_init(model, tmpl_raw);
    check(common_chat_templates_source(tmpls_user.get()) == tmpl_raw, "--chat-template: model template untouched");

    // 找不到那段分支就不改動
    std::string other        = "{% for message in messages %}{{ message.role }}{% endfor %}";
    std::string other_before = other;
    check(!densex_chat_template_fix_late_system(other), "template without the raise branch: fix not applied");
    check(other == other_before, "template without the raise branch: content unchanged");
    std::string raw_copy = tmpl_raw;
    check(densex_chat_template_fix_late_system(raw_copy), "model template: fix applied");
    check(raw_copy == fixed_file, "model template: patched text == fixed reference file");

    // 非數字值跟其他 DENSEX_* 開關一樣：atoi() 算 0 → 關閉（log 會印出實際值，不會假裝是 0）
    common_set_env("DENSEX_LATE_SYSTEM", "true");
    check(!densex_chat_template_late_system_enabled(),
          "DENSEX_LATE_SYSTEM=true is off under atoi() rules, same as the other switches");
    auto tmpls_nonnum = common_chat_templates_init(model, "");
    check(common_chat_templates_source(tmpls_nonnum.get()) == tmpl_raw,
          "non-numeric value: template unchanged and the log shows DENSEX_LATE_SYSTEM=true");
    unset_late_system_env();

    std::cout << "\n=== 2. normal shapes: shipped template vs auto-patched, byte for byte ===\n";
    size_t compared = 0;
    for (const auto & shape : normal_shapes()) {
        for (const auto & kw : kwargs_cases()) {
            auto make_inputs = [&]() {
                common_chat_templates_inputs inputs;
                inputs.messages              = shape.messages;
                inputs.tools                 = shape.tools;
                inputs.add_generation_prompt = shape.add_generation_prompt;
                inputs.enable_thinking       = kw.enable_thinking;
                inputs.chat_template_kwargs  = kw.kwargs;
                return inputs;
            };
            const std::string before = render(tmpls_off.get(), make_inputs());
            const std::string after  = render(tmpls_on.get(), make_inputs());
            ++compared;
            check(before == after && !before.empty(),
                  shape.name + " / " + kw.name + ": identical (" + std::to_string(after.size()) + " bytes)");
            if (!shape.add_generation_prompt) {
                // 真的以 add_generation_prompt=false 渲染：結尾是某個回合的 <|im_end|>\n，而且跟 true 的渲染不同
                auto with_gen        = make_inputs();
                with_gen.add_generation_prompt = true;
                const std::string with_gen_out = render(tmpls_on.get(), with_gen);
                const bool ends_closed = after.size() >= 11 &&
                                         after.compare(after.size() - 11, 11, "<|im_end|>\n") == 0;
                check(ends_closed && with_gen_out != after && with_gen_out.size() > after.size(),
                      shape.name + " / " + kw.name + ": rendered with add_generation_prompt=false ("
                          + std::to_string(after.size()) + " vs " + std::to_string(with_gen_out.size())
                          + " bytes when true)");
            }
        }
    }

    std::cout << "\n=== 3. system messages in the middle of the conversation ===\n";
    struct late_case {
        std::string                   name;
        std::vector<common_chat_msg>  messages;
        std::vector<common_chat_tool> tools;
        std::vector<std::string>      markers; // 每一段中途 system 的內容（也是該回合應該出現的 content）
    };
    std::vector<late_case> late_cases;

    late_cases.push_back({ "L1_between_user_and_assistant",
                           { msg("system", "Opening rules."),
                             msg("user", "First question"),
                             msg("system", "LATE MARKER 1"),
                             msg("assistant", "First answer") },
                           {}, { "LATE MARKER 1" } });

    late_cases.push_back({ "L2_after_tool_result",
                           { msg("user", "Weather in Taipei?"),
                             [] {
                                 common_chat_msg call = msg("assistant", "");
                                 call.tool_calls      = { { "get_weather", R"({"city":"Taipei"})", "call00001" } };
                                 return call;
                             }(),
                             [] {
                                 common_chat_msg res  = msg("tool", "Temperature: 27C");
                                 res.tool_name        = "get_weather";
                                 res.tool_call_id     = "call00001";
                                 return res;
                             }(),
                             msg("system", "LATE MARKER 2"),
                             msg("user", "And tomorrow?") },
                           one_tool(), { "LATE MARKER 2" } });

    late_cases.push_back({ "L3_last_before_generation_prompt",
                           { msg("system", "Opening rules."), msg("user", "Hello"), msg("assistant", "Hi"),
                             msg("system", "LATE MARKER 3") },
                           {}, { "LATE MARKER 3" } });

    late_cases.push_back({ "L4_two_late_systems",
                           { msg("system", "Opening rules."), msg("user", "One"),
                             msg("system", "LATE MARKER A"), msg("assistant", "Two"),
                             msg("system", "LATE MARKER B"), msg("user", "Three") },
                           {}, { "LATE MARKER A", "LATE MARKER B" } });

    // 兩個 tool 結果中間夾一則 system（agy 審查補的形狀）
    {
        common_chat_msg call = msg("assistant", "");
        call.tool_calls      = { { "get_weather", R"({"city":"Taipei"})", "call00001" },
                                 { "get_weather", R"({"city":"Kaohsiung"})", "call00002" } };
        common_chat_msg r1   = msg("tool", "Temperature: 27C");
        r1.tool_name         = "get_weather";
        r1.tool_call_id      = "call00001";
        common_chat_msg r2   = msg("tool", "Temperature: 30C");
        r2.tool_name         = "get_weather";
        r2.tool_call_id      = "call00002";
        late_cases.push_back({ "L5_tool_system_tool",
                               { msg("user", "Weather in Taipei and Kaohsiung?"), call, r1,
                                 msg("system", "LATE MARKER 5"), r2, msg("user", "Thanks") },
                               one_tool(), { "LATE MARKER 5" } });
    }

    // 第二則是 developer（Codex 的實際形狀）：引擎在 apply 裡把 developer 對應成 system，所以是中途 system
    late_cases.push_back({ "L6_developer_second",
                           { msg("system", "Opening rules."),
                             msg("developer", "LATE MARKER DEV"),
                             msg("user", "Hello") },
                           {}, { "LATE MARKER DEV" } });

    // 空白的中途 system：不該輸出空回合（跟開頭 system 的 {%- if content %} 守門一致）
    late_cases.push_back({ "L7_blank_late_system_dropped",
                           { msg("system", "Opening rules."),
                             msg("user", "Hello"),
                             msg("system", "   \n  "),
                             msg("assistant", "Hi") },
                           {}, {} });

    size_t late_checked = 0;
    for (const auto & c : late_cases) {
        for (const auto & kw : kwargs_cases()) {
            auto make_inputs = [&]() {
                common_chat_templates_inputs inputs;
                inputs.messages              = c.messages;
                inputs.tools                 = c.tools;
                inputs.add_generation_prompt = true;
                inputs.chat_template_kwargs  = kw.kwargs;
                inputs.enable_thinking       = kw.enable_thinking;
                return inputs;
            };
            const std::string label = c.name + " / " + kw.name;
            ++late_checked;

            const std::string err_off = render_error(tmpls_off.get(), make_inputs());
            check(err_off.find("System message must be at the beginning.") != std::string::npos,
                  label + ": shipped template still rejects it (DENSEX_LATE_SYSTEM=0)");

            const std::string err_on = render_error(tmpls_on.get(), make_inputs());
            check(err_on.empty(), label + ": patched template renders without error");
            if (!err_on.empty()) {
                continue;
            }
            const std::string out = render(tmpls_on.get(), make_inputs());

            int turns = 0;
            for (const auto & marker : c.markers) {
                const int n = count_full_system_turns(out, marker);
                check(n == 1, label + ": '" + marker + "' rendered as exactly one own system turn closed by <|im_end|>\\n");
                turns += n;
            }
            check(!has_empty_system_turn(out), label + ": no empty system turn in the prompt");
            if (c.markers.empty()) {
                // 沒有中途 system 回合：整個 prompt 只剩開頭那一則 system（這個案例沒帶工具，所以 tools 區塊不會出現）
                check(count_sub(out, "<|im_start|>system\n") == 1,
                      label + ": blank late system produced no turn (opening system kept)");
            }
        }
    }

    std::cout << "\n=== 4. image inside a late system message ===\n";
    {
        const std::string input = R"({
            "messages": [
                { "role": "user", "content": "Hi" },
                { "role": "system", "content": [
                    { "type": "image", "image": "https://example.invalid/x.png" }
                ] },
                { "role": "user", "content": "What is in the picture?" }
            ],
            "add_generation_prompt": true,
            "enable_thinking": false
        })";
        const std::string err = render_direct_jinja_error(common_chat_templates_source(tmpls_on.get()), input);
        check(err.find("System message cannot contain images.") != std::string::npos,
              "late system with an image raises, like the first system message");
    }

    std::cout << "\n=== 5. the comparison above really catches a stray newline ===\n";
    {
        std::string mutated = tmpl_raw;
        const std::string user_branch =
            "    {%- elif message.role == \"user\" %}\n"
            "        {{- '<|im_start|>' + message.role + '\\n' + content + '<|im_end|>' + '\\n' }}\n";
        const std::string user_branch_extra_nl =
            "    {%- elif message.role == \"user\" %}\n"
            "        {{- '<|im_start|>' + message.role + '\\n' + content + '<|im_end|>' + '\\n' + '\\n' }}\n";
        if (mutated.find(user_branch) == std::string::npos) {
            check(false, "mutation target (user turn output) found in the model template");
        } else {
            string_replace_all(mutated, user_branch, user_branch_extra_nl);
            check(densex_chat_template_fix_late_system(mutated),
                  "mutated template still matches the late-system branch");
            auto tmpls_mut = common_chat_templates_init(nullptr, mutated);
            const auto shapes = normal_shapes();
            common_chat_templates_inputs inputs;
            inputs.messages              = shapes[2].messages; // 03_multi_turn
            inputs.add_generation_prompt = true;
            inputs.chat_template_kwargs  = kwargs_off();
            inputs.enable_thinking       = false;
            const std::string baseline = render(tmpls_on.get(), inputs);
            const std::string broken   = render(tmpls_mut.get(), inputs);
            check(baseline != broken, "a normal shape with one extra newline is reported as a difference");
        }
    }

    llama_model_free(model);
    llama_backend_free();

    std::cout << "\n=== SUMMARY ===\n";
    std::cout << "byte comparisons of normal shapes: " << compared / 2 << " shapes x " << kwargs_cases().size()
              << " kwargs groups = " << compared << " comparisons\n";
    std::cout << "late system cases x kwargs groups: " << late_checked << "\n";
    if (failures) {
        std::cout << "QWEN38_LATE_SYSTEM_FAILED failures=" << failures << "\n";
        return 1;
    }
    std::cout << "QWEN38_LATE_SYSTEM_OK\n";
    return 0;
}
