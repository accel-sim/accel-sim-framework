/*
 * Generic basic-block pattern matcher for NVBit-based tools.
 *
 * The library knows about NVBit's `Instr` and `basic_block_t` / `CFG_t`
 * types. It is intentionally domain-agnostic: callers register their own
 * named predicates over a BB's instruction list with addPattern(), then
 * apply them via matchBB() / matchCFG(). No built-in patterns are shipped.
 *
 * The spinlock detection tool is one consumer; other NVBit tools can reuse
 * this header by including it directly.
 */

#ifndef BB_PATTERN_MATCHER_H
#define BB_PATTERN_MATCHER_H

#include <functional>
#include <map>
#include <string>
#include <vector>

#include "nvbit.h"

/* A named predicate over a basic block. */
struct BBPattern {
    std::string name;
    std::function<bool(const std::vector<Instr*>&)> match;
};

class BBPatternMatcher {
public:
    BBPatternMatcher() = default;

    /* Register a pattern. Patterns are applied in registration order. */
    void addPattern(BBPattern p);
    void addPattern(std::string name,
                    std::function<bool(const std::vector<Instr*>&)> match);

    /* Apply every registered pattern to one BB. Returns the names of the
     * patterns that matched (empty vector if none). */
    std::vector<std::string> matchBB(
        const std::vector<Instr*>& bb_instrs) const;

    /* Apply to a whole CFG. Returns map { bb_index -> [matched names] }.
     * Degenerate CFGs (cfg.is_degenerate == true) yield an empty map. */
    std::map<size_t, std::vector<std::string>> matchCFG(
        const CFG_t& cfg) const;

    size_t size() const { return patterns_.size(); }

private:
    std::vector<BBPattern> patterns_;
};

/* --- Free helpers usable inside any BBPattern::match body --- */

/* True if `instr->getOpcode()` (full dotted mnemonic, e.g.
 * "SYNCS.PHASECHK.TRANS64.TRYWAIT") contains `needle` as a substring.
 * Takes a non-const Instr* because NVBit's accessors aren't const-qualified. */
bool opcode_contains(Instr* instr, const std::string& needle);

/* True if any instruction in `bb_instrs` has an opcode containing `needle`. */
bool bb_has_opcode_substring(const std::vector<Instr*>& bb_instrs,
                             const std::string& needle);

/* True if `instr->getOpcodeShort()` equals `mnemonic` exactly. */
bool opcode_short_equals(Instr* instr, const std::string& mnemonic);

#endif  // BB_PATTERN_MATCHER_H
