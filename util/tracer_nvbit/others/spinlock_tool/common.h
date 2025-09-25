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
#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
#include <regex>

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