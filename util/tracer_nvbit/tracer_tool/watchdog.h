#ifndef WATCHDOG_H
#define WATCHDOG_H

#include <assert.h>
#include <memory>
#include <stdint.h>
#include <string.h>

/* nvbit interface file for Instr class */
#include "nvbit.h"

/**
 * Pure virtual interface for watchdog implementations
 * */
class WatchdogInterface {
public:
  virtual ~WatchdogInterface() = default;
  virtual void reset() = 0;
  virtual bool is_in_region() const = 0;
  virtual void observe_instruction(Instr *const instr) = 0;
};

/**
 * Implementation that skips instructions between
 * WARPSYNC.COLLECTIVE and its target instruction (inclusive)
 * */
class WarpsyncCollectiveWatchdog : public WatchdogInterface {
private:
  bool in_region = false; // true if we are in the region of interest
  uint64_t start_pc = 0;  // PC of the WARPSYNC.COLLECTIVE instruction
  uint64_t end_pc = 0;    // PC of the target instruction

public:
  void reset() override {
    in_region = false;
    start_pc = 0;
    end_pc = 0;
  }

  bool is_in_region() const override { return in_region; }

  void observe_instruction(Instr *const instr) override {
    if (in_region) {
      if (instr->getOffset() > end_pc) {
        reset();
      }
    }

    // if opcode is WARPSYNC.COLLECTIVE, set in_region to true
    if (strcmp(instr->getOpcode(), "WARPSYNC.COLLECTIVE") == 0) {
      assert(1 < instr->getNumOperands());
      in_region = true;
      start_pc = instr->getOffset();
      end_pc = instr->getOperand(1)->u.imm_uint64.value;
    }
  }
};

/**
 * Null object implementation that does nothing
 * */
class NullWatchdog : public WatchdogInterface {
public:
  void reset() override {
    // Do nothing
  }

  bool is_in_region() const override { return false; }

  void observe_instruction(Instr *const instr) override {
    // Do nothing
  }
};

/**
 * Factory class for creating watchdog implementations
 * */
class WatchdogFactory {
public:
  /**
   * Creates a watchdog implementation based on the enable flag
   * @param enable_watchdog If true, returns WarpsyncCollectiveWatchdog; if
   * false, returns NullWatchdog
   * @return unique_ptr to WatchdogInterface implementation
   */
  static std::unique_ptr<WatchdogInterface> create(bool enable_watchdog) {
    if (enable_watchdog) {
      return std::make_unique<WarpsyncCollectiveWatchdog>();
    } else {
      return std::make_unique<NullWatchdog>();
    }
  }
};

#endif // WATCHDOG_H
