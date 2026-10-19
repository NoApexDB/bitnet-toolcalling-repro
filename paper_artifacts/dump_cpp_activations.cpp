#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <algorithm>

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

struct TensorDump {
    std::string name;
    int type;
    int64_t ne[4];
    double global_l2;
    double global_mean;
    double global_std;
    double global_min;
    double global_max;
    double last_token_l2;
    double last_token_mean;
    double last_token_std;
    double last_token_min;
    double last_token_max;
    std::vector<float> last_token_first10;
    std::vector<float> last_token_full;
};

static std::vector<TensorDump> g_dumps;

static bool eval_callback(struct ggml_tensor * t, bool ask, void * user_data) {
    if (ask) return true;
    if (!t || !t->data) return true;

    std::string name = t->name ? t->name : "";
    if (name.empty()) return true;

    if (name.find("attn_norm") == std::string::npos &&
        name.find("Qcur") == std::string::npos &&
        name.find("Kcur") == std::string::npos &&
        name.find("Vcur") == std::string::npos &&
        name.find("attn_sub_norm") == std::string::npos &&
        name.find("attn_out") == std::string::npos &&
        name.find("ffn_inp") == std::string::npos &&
        name.find("ffn_norm") == std::string::npos &&
        name.find("ffn_up") == std::string::npos &&
        name.find("ffn_gate") == std::string::npos &&
        name.find("ffn_relu") == std::string::npos &&
        name.find("ffn_sqr") == std::string::npos &&
        name.find("ffn_gate_par") == std::string::npos &&
        name.find("ffn_swiglu") == std::string::npos &&
        name.find("ffn_sub_out") == std::string::npos &&
        name.find("ffn_sub_norm") == std::string::npos &&
        name.find("ffn_down") == std::string::npos &&
        name.find("l_out") == std::string::npos &&
        name.find("result_norm") == std::string::npos &&
        name.find("result_output") == std::string::npos) {
        return true;
    }

    if (t->type != GGML_TYPE_F32) {
        return true;
    }

    TensorDump td;
    td.name = name;
    td.type = t->type;
    td.ne[0] = t->ne[0];
    td.ne[1] = t->ne[1];
    td.ne[2] = t->ne[2];
    td.ne[3] = t->ne[3];

    int64_t n_elems = ggml_nelements(t);
    const float * data = (const float *)t->data;

    double sum = 0.0, sum_sq = 0.0;
    double min_v = data[0], max_v = data[0];
    for (int64_t i = 0; i < n_elems; i++) {
        double v = data[i];
        sum += v;
        sum_sq += v * v;
        if (v < min_v) min_v = v;
        if (v > max_v) max_v = v;
    }
    double mean = sum / n_elems;
    double var = (sum_sq / n_elems) - (mean * mean);
    if (var < 0) var = 0;
    td.global_l2 = sqrt(sum_sq);
    td.global_mean = mean;
    td.global_std = sqrt(var);
    td.global_min = min_v;
    td.global_max = max_v;

    int64_t n_feats = t->ne[0];
    int64_t last_tok_idx = t->ne[1] - 1;
    if (last_tok_idx >= 0 && t->ne[1] > 0) {
        const float * tok_ptr = (const float *)((const char *)t->data + last_tok_idx * t->nb[1]);
        double tok_sum = 0.0, tok_sum_sq = 0.0;
        double tok_min = tok_ptr[0], tok_max = tok_ptr[0];
        for (int64_t f = 0; f < n_feats; f++) {
            double v = tok_ptr[f];
            tok_sum += v;
            tok_sum_sq += v * v;
            if (v < tok_min) tok_min = v;
            if (v > tok_max) tok_max = v;
            if (f < 10) td.last_token_first10.push_back((float)v);
            if (name.find("-0") != std::string::npos || name.find("result_output") != std::string::npos || name.find("result_norm") != std::string::npos) {
                td.last_token_full.push_back((float)v);
            }
        }
        double tok_mean = tok_sum / n_feats;
        double tok_var = (tok_sum_sq / n_feats) - (tok_mean * tok_mean);
        if (tok_var < 0) tok_var = 0;
        td.last_token_l2 = sqrt(tok_sum_sq);
        td.last_token_mean = tok_mean;
        td.last_token_std = sqrt(tok_var);
        td.last_token_min = tok_min;
        td.last_token_max = tok_max;
    }

    g_dumps.push_back(td);
    return true;
}

std::vector<llama_token> load_tokens_from_json(const std::string & filename) {
    std::ifstream f(filename);
    if (!f.is_open()) {
        fprintf(stderr, "Cannot open tokens JSON: %s\n", filename.c_str());
        exit(1);
    }
    std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    size_t pos = s.find("\"tokens\":");
    if (pos == std::string::npos) {
        fprintf(stderr, "Missing 'tokens' key in JSON\n");
        exit(1);
    }
    size_t start = s.find('[', pos);
    size_t end = s.find(']', start);
    std::string arr = s.substr(start + 1, end - start - 1);

    std::vector<llama_token> tokens;
    char * ptr = &arr[0];
    char * token_str = strtok(ptr, " ,\r\n\t");
    while (token_str) {
        tokens.push_back((llama_token)atoi(token_str));
        token_str = strtok(NULL, " ,\r\n\t");
    }
    return tokens;
}

static std::string escape_json(const std::string & s) {
    std::string out;
    for (char c : s) {
        if (c == '"') out += "\\\"";
        else if (c == '\\') out += "\\\\";
        else if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else if (c == '\t') out += "\\t";
        else out += c;
    }
    return out;
}

int main(int argc, char ** argv) {
    std::string model_path = "C:\\ai_models\\BitNet\\models\\BitNet-b1.58-2B-4T\\ggml-model-i2_s.gguf";
    std::string tokens_json = "C:\\ai_models\\paper_artifacts\\diagnostic_input_tokens.json";
    std::string out_json = "C:\\ai_models\\paper_artifacts\\cpp_layer_activations.json";
    int n_threads = 4;

    if (argc > 1) model_path = argv[1];
    if (argc > 2) tokens_json = argv[2];
    if (argc > 3) out_json = argv[3];
    if (argc > 4) n_threads = atoi(argv[4]);

    printf("=== DUMPING BITNET.CPP INTERMEDIATE ACTIVATIONS ===\n");
    printf("Model: %s\n", model_path.c_str());
    printf("Tokens: %s\n", tokens_json.c_str());
    printf("Output: %s\n", out_json.c_str());
    printf("Threads: %d\n", n_threads);

    std::vector<llama_token> tokens = load_tokens_from_json(tokens_json);
    int32_t n_tokens = (int32_t)tokens.size();
    printf("Loaded %d input tokens.\n", n_tokens);

    ggml_backend_load_all();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0; // CPU only

    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) {
        fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    int32_t n_vocab = llama_vocab_n_tokens(vocab);
    printf("Model loaded. Vocab size: %d\n", n_vocab);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = n_tokens + 16;
    cparams.n_batch = n_tokens;
    cparams.n_threads = n_threads;
    cparams.n_threads_batch = n_threads;
    cparams.cb_eval = eval_callback;
    cparams.cb_eval_user_data = nullptr;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "Failed to init context\n");
        return 1;
    }

    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    for (int32_t i = 0; i < n_tokens; i++) {
        batch.token[i] = tokens[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = (i == n_tokens - 1);
    }
    batch.n_tokens = n_tokens;

    printf("Decoding batch of %d tokens...\n", n_tokens);
    int res = llama_decode(ctx, batch);
    if (res != 0) {
        fprintf(stderr, "llama_decode failed with code %d\n", res);
        return 1;
    }
    printf("Decode finished successfully! Captured %zu tensor evaluations.\n", g_dumps.size());

    // Extract logits for last token
    float * logits = llama_get_logits_ith(ctx, n_tokens - 1);
    if (!logits) {
        fprintf(stderr, "Failed to get logits\n");
        return 1;
    }

    // Compute logits statistics
    double logit_sum = 0.0, logit_sum_sq = 0.0;
    double logit_min = logits[0], logit_max = logits[0];
    std::vector<std::pair<float, int32_t>> scored_tokens;
    scored_tokens.reserve(n_vocab);

    for (int32_t i = 0; i < n_vocab; i++) {
        float l = logits[i];
        logit_sum += l;
        logit_sum_sq += (double)l * l;
        if (l < logit_min) logit_min = l;
        if (l > logit_max) logit_max = l;
        scored_tokens.push_back({l, i});
    }

    double logit_mean = logit_sum / n_vocab;
    double logit_var = (logit_sum_sq / n_vocab) - (logit_mean * logit_mean);
    double logit_std = sqrt(logit_var > 0 ? logit_var : 0);
    double logit_l2 = sqrt(logit_sum_sq);

    std::sort(scored_tokens.begin(), scored_tokens.end(), [](const auto & a, const auto & b) {
        return a.first > b.first;
    });

    // Softmax for top 10
    double max_l = scored_tokens[0].first;
    double exp_sum = 0.0;
    for (int i = 0; i < n_vocab; i++) {
        exp_sum += exp(logits[i] - max_l);
    }

    printf("\n--- CPP Top 5 Next Token Predictions ---\n");
    for (int i = 0; i < 5; i++) {
        int32_t tid = scored_tokens[i].second;
        float logit = scored_tokens[i].first;
        double prob = exp(logit - max_l) / exp_sum;
        char piece_buf[128] = {0};
        llama_token_to_piece(vocab, tid, piece_buf, sizeof(piece_buf), 0, true);
        printf("  Rank %d: ID %6d | logit: %8.4f | prob: %6.2f%% | piece: '%s'\n",
               i + 1, tid, logit, prob * 100.0, piece_buf);
    }

    // Save JSON
    std::ofstream out(out_json);
    out << std::setprecision(6) << std::fixed;
    out << "{\n";
    out << "  \"n_tokens\": " << n_tokens << ",\n";
    out << "  \"total_tensors_captured\": " << g_dumps.size() << ",\n";
    out << "  \"logits_stats\": {\n";
    out << "    \"l2\": " << logit_l2 << ",\n";
    out << "    \"mean\": " << logit_mean << ",\n";
    out << "    \"std\": " << logit_std << ",\n";
    out << "    \"min\": " << logit_min << ",\n";
    out << "    \"max\": " << logit_max << ",\n";
    out << "    \"full_logits\": [";
    for (int i = 0; i < n_vocab; i++) {
        out << logits[i];
        if (i + 1 < n_vocab) out << ", ";
    }
    out << "],\n";
    out << "    \"top10\": [\n";
    for (int i = 0; i < 10 && i < n_vocab; i++) {
        int32_t tid = scored_tokens[i].second;
        float logit = scored_tokens[i].first;
        double prob = exp(logit - max_l) / exp_sum;
        char piece_buf[128] = {0};
        llama_token_to_piece(vocab, tid, piece_buf, sizeof(piece_buf), 0, true);
        out << "      {\"rank\": " << (i+1) << ", \"id\": " << tid
            << ", \"logit\": " << logit << ", \"prob\": " << prob
            << ", \"piece\": \"" << escape_json(piece_buf) << "\"}";
        if (i < 9) out << ",";
        out << "\n";
    }
    out << "    ]\n";
    out << "  },\n";
    out << "  \"tensors\": [\n";
    for (size_t i = 0; i < g_dumps.size(); i++) {
        const auto & td = g_dumps[i];
        out << "    {\n";
        out << "      \"name\": \"" << td.name << "\",\n";
        out << "      \"type\": " << td.type << ",\n";
        out << "      \"ne\": [" << td.ne[0] << ", " << td.ne[1] << ", " << td.ne[2] << ", " << td.ne[3] << "],\n";
        out << "      \"global_l2\": " << td.global_l2 << ",\n";
        out << "      \"global_mean\": " << td.global_mean << ",\n";
        out << "      \"global_std\": " << td.global_std << ",\n";
        out << "      \"last_tok_l2\": " << td.last_token_l2 << ",\n";
        out << "      \"last_tok_mean\": " << td.last_token_mean << ",\n";
        out << "      \"last_tok_std\": " << td.last_token_std << ",\n";
        out << "      \"last_tok_first10\": [";
        for (size_t j = 0; j < td.last_token_first10.size(); j++) {
            out << td.last_token_first10[j];
            if (j + 1 < td.last_token_first10.size()) out << ", ";
        }
        out << "]";
        if (!td.last_token_full.empty()) {
            out << ",\n      \"last_token_full\": [";
            for (size_t j = 0; j < td.last_token_full.size(); j++) {
                out << td.last_token_full[j];
                if (j + 1 < td.last_token_full.size()) out << ", ";
            }
            out << "]";
        }
        out << "\n    }";
        if (i + 1 < g_dumps.size()) out << ",";
        out << "\n";
    }
    out << "  ]\n";
    out << "}\n";

    printf("\nWrote %zu tensor entries to: %s\n", g_dumps.size(), out_json.c_str());

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);

    return 0;
}
