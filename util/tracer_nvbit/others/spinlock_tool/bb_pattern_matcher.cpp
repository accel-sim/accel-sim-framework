#include "bb_pattern_matcher.h"

#include <cstring>
#include <utility>

void BBPatternMatcher::addPattern(BBPattern p) {
    patterns_.push_back(std::move(p));
}

void BBPatternMatcher::addPattern(
    std::string name,
    std::function<bool(const std::vector<Instr*>&)> match) {
    patterns_.push_back(BBPattern{std::move(name), std::move(match)});
}

std::vector<std::string> BBPatternMatcher::matchBB(
    const std::vector<Instr*>& bb_instrs) const {
    std::vector<std::string> hits;
    for (const auto& p : patterns_) {
        if (p.match && p.match(bb_instrs)) {
            hits.push_back(p.name);
        }
    }
    return hits;
}

std::map<size_t, std::vector<std::string>> BBPatternMatcher::matchCFG(
    const CFG_t& cfg) const {
    std::map<size_t, std::vector<std::string>> out;
    if (cfg.is_degenerate) {
        return out;
    }
    for (size_t bb_id = 0; bb_id < cfg.bbs.size(); bb_id++) {
        auto* bb = cfg.bbs[bb_id];
        if (!bb) continue;
        auto hits = matchBB(bb->instrs);
        if (!hits.empty()) {
            out[bb_id] = std::move(hits);
        }
    }
    return out;
}

bool opcode_contains(Instr* instr, const std::string& needle) {
    if (!instr) return false;
    const char* op = instr->getOpcode();
    if (!op) return false;
    return std::strstr(op, needle.c_str()) != nullptr;
}

bool bb_has_opcode_substring(const std::vector<Instr*>& bb_instrs,
                             const std::string& needle) {
    for (Instr* i : bb_instrs) {
        if (opcode_contains(i, needle)) return true;
    }
    return false;
}

bool opcode_short_equals(Instr* instr, const std::string& mnemonic) {
    if (!instr) return false;
    const char* op = instr->getOpcodeShort();
    if (!op) return false;
    return mnemonic == op;
}
