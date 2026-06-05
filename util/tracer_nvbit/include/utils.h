#ifndef _UTILS_H_
#define _UTILS_H_

#include <cstddef>
#include <functional>
#include <tuple>

namespace accelsimUtils {
  // Generic hash for std::tuple so tuples can be used as keys in unordered
  // associative containers (e.g. std::unordered_set/map).
  struct TupleHash {
    template <class... Ts>
    std::size_t operator()(const std::tuple<Ts...>& t) const {
      std::size_t seed = 0;
      std::apply(
          [&seed](const auto&... elems) { (hash_combine(seed, elems), ...); },
          t);
      return seed;
    }

   private:
    template <class T>
    static void hash_combine(std::size_t& seed, const T& value) {
      // Boost-style hash combine for a better distribution than a plain XOR.
      seed ^= std::hash<T>{}(value) + 0x9e3779b97f4a7c15ULL + (seed << 6) +
              (seed >> 2);
    }
  };
}  // namespace accelsimUtils

#endif
