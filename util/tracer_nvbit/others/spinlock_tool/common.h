/*
 * SPDX-FileCopyrightText: Copyright (c) 2019 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <stdint.h>
#include <map>
#include <set>
#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
#include <regex>
#include <utility>
#include <vector>

/* information collected in the instrumentation function and passed
 * on the channel from the GPU to the CPU */
typedef struct {
    uint32_t instr_idx;
    uint32_t count;
} instr_count_t;


/* Class to hold kernel instruction histogram */
class KernelInstructionHistogram {
public:
    KernelInstructionHistogram() 
        : id(0), name("dummy"), histogram(std::map<uint32_t, uint32_t>()) {
    }

    KernelInstructionHistogram(uint32_t id, std::string name) 
        : id(id), name(name), histogram(std::map<uint32_t, uint32_t>()) {
    }

    void add(uint32_t instr_idx, uint64_t count) {
        if (histogram.find(instr_idx) == histogram.end()) {
            histogram[instr_idx] = count;
        } else {
            histogram[instr_idx] += count;
        }
    }

    void merge(const KernelInstructionHistogram& other, bool use_hash = false) {
        for (const auto& [instr_idx, count] : other.histogram) {
            if (use_hash) {
                // Simple modulo hash operation
                add(instr_idx, count % hash_prime);
                histogram[instr_idx] %= hash_prime;
            } else {
                add(instr_idx, count);
            }
        }
    }

    void reinit(uint32_t id, std::string name) {
        this->id = id;
        this->name = name;
        histogram.clear();
    }

    std::map<uint32_t, std::pair<uint32_t, uint32_t>> findSpinlock(const KernelInstructionHistogram& other) {
        // Find instructions that have different execution counts between two runs
        // These are likely part of spinlock sections
        std::map<uint32_t, std::pair<uint32_t, uint32_t>> spinlockInstructions;
        
        // Check all instructions in this histogram
        for (const auto& [instrIdx, count] : histogram) {
            auto otherIt = other.histogram.find(instrIdx);
            if (otherIt != other.histogram.end()) {
                // Instruction exists in both histograms
                if (count != otherIt->second) {
                    // Different execution counts - likely spinlock
                    spinlockInstructions[instrIdx] = {count, otherIt->second};
                }
            } else {
                // Instruction only exists in this histogram
                spinlockInstructions[instrIdx] = {count, 0};
            }
        }
        
        // Check instructions that only exist in the other histogram
        for (const auto& [instrIdx, count] : other.histogram) {
            if (histogram.find(instrIdx) == histogram.end()) {
                // Instruction only exists in other histogram
                spinlockInstructions[instrIdx] = {0, count}; // Mark as 0 in this run
            }
        }
        
        return spinlockInstructions;
    }

    // Save histogram to file
    bool saveToFile(const std::string& filename) const {
        std::ofstream file(filename);
        if (!file.is_open()) {
            return false;
        }
        file << serialize();
        file.close();
        return true;
    }

    // Load histogram from file
    bool loadFromFile(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            return false;
        }
        
        std::stringstream buffer;
        buffer << file.rdbuf();
        file.close();
        
        deserialize(buffer.str());
        return true;
    }

    // Get total instruction count
    uint64_t getTotalInstructionCount() const {
        uint64_t total = 0;
        for (const auto& [instrIdx, count] : histogram) {
            total += count;
        }
        return total;
    }

    // Get number of unique instructions
    size_t getUniqueInstructionCount() const {
        return histogram.size();
    }

    // Check if histogram is empty
    bool isEmpty() const {
        return histogram.empty();
    }

    // Clear histogram
    void clear() {
        histogram.clear();
    }

    std::string serialize() const {
        std::stringstream ss;
        ss << "Kernel: " << name << " (ID: " << id << ")" << std::endl;
        for (const auto &[instr_idx, count] : histogram) {
            ss << instr_idx << ": " << count << std::endl;
        }
        return ss.str();
    }

    void deserialize(const std::string& data) {
        // Deserialize the data following the serialize format
        // Kernel: <name> (ID: <id>)
        // <instr_idx>: <count>
        // <instr_idx>: <count>
        // ...
        std::stringstream ss(data);
        std::string line;
        
        // Clear existing histogram
        histogram.clear();
        
        // Regex patterns for parsing
        std::regex headerPattern(R"(Kernel:\s*(.+?)\s*\(ID:\s*(\d+)\))");
        std::regex instructionPattern(R"(\s*(\d+)\s*:\s*(\d+)\s*)");
        
        // Parse header line: "Kernel: <name> (ID: <id>)"
        if (std::getline(ss, line)) {
            std::smatch headerMatch;
            if (std::regex_match(line, headerMatch, headerPattern)) {
                if (headerMatch.size() >= 3) {
                    name = headerMatch[1].str();
                    id = std::stoul(headerMatch[2].str());
                }
            }
        }
        
        // Parse instruction count lines: "<instr_idx>: <count>"
        while (std::getline(ss, line)) {
            if (line.empty()) continue;
            
            std::smatch instructionMatch;
            if (std::regex_match(line, instructionMatch, instructionPattern)) {
                if (instructionMatch.size() >= 3) {
                    try {
                        uint32_t instrIdx = std::stoul(instructionMatch[1].str());
                        uint32_t count = std::stoul(instructionMatch[2].str());
                        histogram[instrIdx] = count;
                    } catch (const std::exception& e) {
                        // Skip malformed lines
                        continue;
                    }
                }
            }
        }
    }

    uint32_t id;
    std::string name;
    std::map<uint32_t, uint32_t> histogram;
    // A large 30-bit prime number for hashing to avoid overflow
    static constexpr uint32_t hash_prime = 1073741789;
};

/* Static basic-block layout of a kernel function, captured from
 * NVBit's CFG at instrumentation time. Used by the spinlock detector
 * to promote per-instruction count-diffs to whole-BB regions. */
class KernelBBLayout {
public:
    KernelBBLayout() : id(0), name("dummy"), is_degenerate(false) {}

    bool saveToFile(const std::string& filename) const {
        std::ofstream file(filename);
        if (!file.is_open()) return false;
        file << serialize();
        file.close();
        return true;
    }

    bool loadFromFile(const std::string& filename) {
        std::ifstream file(filename);
        if (!file.is_open()) return false;
        std::stringstream buffer;
        buffer << file.rdbuf();
        file.close();
        return deserialize(buffer.str());
    }

    std::string serialize() const {
        std::stringstream ss;
        // Reuse the histogram header convention so both file types share
        // tooling. The trailing "Degenerate:" line distinguishes a layout
        // file from a histogram file.
        ss << "Kernel: " << name << " (ID: " << id << ")" << std::endl;
        ss << "Degenerate: " << (is_degenerate ? 1 : 0) << std::endl;
        for (size_t bb_id = 0; bb_id < bbs.size(); bb_id++) {
            ss << "BB " << bb_id << ":";
            for (uint32_t idx : bbs[bb_id]) {
                ss << " " << idx;
            }
            ss << std::endl;
        }
        // Per-BB static-pattern hits, keyed by BB id. Written after the
        // BB lines so older parsers ignoring "Pattern" lines still work.
        for (const auto& [bb_id, names] : bb_pattern_names) {
            ss << "Pattern " << bb_id << ":";
            for (const std::string& n : names) {
                ss << " " << n;
            }
            ss << std::endl;
        }
        return ss.str();
    }

    /* Returns true on success. A "successful" parse requires that the header
     * line ("Kernel: <name> (ID: <id>)") was matched; without it we'd end
     * up keying the layout under the default name "dummy", silently
     * disabling BB promotion downstream. The remaining body lines are
     * dispatched by regex prefix (Degenerate / BB / Pattern) rather than
     * positionally, so a missing or out-of-order Degenerate line no longer
     * eats the first BB. */
    bool deserialize(const std::string& data) {
        std::stringstream ss(data);
        std::string line;
        bbs.clear();
        static_pattern_bbs.clear();
        bb_pattern_names.clear();
        is_degenerate = false;

        std::regex headerPattern(R"(Kernel:\s*(.+?)\s*\(ID:\s*(\d+)\))");
        std::regex degeneratePattern(R"(Degenerate:\s*(\d+))");
        std::regex bbPattern(R"(BB\s+(\d+)\s*:\s*(.*))");
        std::regex patternPattern(R"(Pattern\s+(\d+)\s*:\s*(.*))");

        bool header_ok = false;
        if (std::getline(ss, line)) {
            std::smatch m;
            if (std::regex_match(line, m, headerPattern) && m.size() >= 3) {
                name = m[1].str();
                id = std::stoul(m[2].str());
                header_ok = true;
            }
        }
        if (!header_ok) return false;

        while (std::getline(ss, line)) {
            if (line.empty()) continue;
            std::smatch m;
            if (std::regex_match(line, m, degeneratePattern) && m.size() >= 2) {
                is_degenerate = (std::stoul(m[1].str()) != 0);
                continue;
            }
            if (std::regex_match(line, m, bbPattern) && m.size() >= 3) {
                std::vector<uint32_t> idxs;
                std::stringstream idx_ss(m[2].str());
                uint32_t idx;
                while (idx_ss >> idx) {
                    idxs.push_back(idx);
                }
                bbs.push_back(std::move(idxs));
                continue;
            }
            if (std::regex_match(line, m, patternPattern) && m.size() >= 3) {
                size_t bb_id = std::stoul(m[1].str());
                std::vector<std::string> names;
                std::stringstream name_ss(m[2].str());
                std::string n;
                while (name_ss >> n) {
                    names.push_back(n);
                }
                if (!names.empty()) {
                    static_pattern_bbs.insert(bb_id);
                    bb_pattern_names[bb_id] = std::move(names);
                }
                continue;
            }
        }
        return true;
    }

    /* Returns the BB id containing instr_idx, or -1 if not found. */
    int findBB(uint32_t instr_idx) const {
        for (size_t i = 0; i < bbs.size(); i++) {
            for (uint32_t idx : bbs[i]) {
                if (idx == instr_idx) return static_cast<int>(i);
            }
        }
        return -1;
    }

    /* Given a set of count-differing instruction indices, return the union
     * of all instructions belonging to any BB that contains at least one
     * of them. Falls back to the input keys verbatim when the CFG was
     * degenerate at capture time. */
    std::set<uint32_t> promoteFromDiffs(
        const std::map<uint32_t, std::pair<uint32_t, uint32_t>>& diffs) const {
        std::set<uint32_t> result;
        if (is_degenerate) {
            for (const auto& [idx, _] : diffs) result.insert(idx);
            return result;
        }
        std::set<size_t> tainted_bbs;
        for (const auto& [idx, _] : diffs) {
            int bb = findBB(idx);
            if (bb >= 0) {
                tainted_bbs.insert(static_cast<size_t>(bb));
            } else {
                // Instr not in any captured BB (shouldn't happen if the
                // layout is current, but be defensive): emit it raw.
                result.insert(idx);
            }
        }
        for (size_t bb_id : tainted_bbs) {
            for (uint32_t idx : bbs[bb_id]) {
                result.insert(idx);
            }
        }
        return result;
    }

    /* Structural equality: same is_degenerate flag, identical BB list, and
     * identical static-pattern flagging (the matched set is a property of
     * the static SASS, so two captures of the same kernel must agree). */
    bool equivalent(const KernelBBLayout& other) const {
        return is_degenerate == other.is_degenerate
            && bbs == other.bbs
            && static_pattern_bbs == other.static_pattern_bbs;
    }

    /* Returns the union of all instruction indices belonging to any BB in
     * static_pattern_bbs. Empty when degenerate or no pattern matched. */
    std::set<uint32_t> staticPatternFlagged() const {
        std::set<uint32_t> result;
        if (is_degenerate) return result;
        for (size_t bb_id : static_pattern_bbs) {
            if (bb_id >= bbs.size()) continue;
            for (uint32_t idx : bbs[bb_id]) {
                result.insert(idx);
            }
        }
        return result;
    }

    uint32_t id;
    std::string name;
    bool is_degenerate;
    std::vector<std::vector<uint32_t>> bbs;
    /* BB ids flagged by the static pattern matcher at instrumentation
     * time. Independent of the count-diff signal. */
    std::set<size_t> static_pattern_bbs;
    /* Per-BB list of pattern names that matched (for warning text). */
    std::map<size_t, std::vector<std::string>> bb_pattern_names;
};