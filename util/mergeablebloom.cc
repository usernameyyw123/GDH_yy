//
// Created by yyw on 2024/1/26.
//
#include "../util/mergeablebloom.h"
#include "../util/hash.h"
namespace  leveldb {

bool MergeableBloom::KeyMayMatch(leveldb::Slice& key) {
  if (reslut_size < 8) return false;

  const size_t bits = reslut_size * 8;

  uint32_t h = BloomHash(key);
  const uint32_t delta = (h >> 17) | (h << 15);  // Rotate right 17 bits
  for (size_t j = 0; j < k_; j++) {
    const uint32_t bitpos = h % bits;
    if ((result[bitpos / 8] & (1 << (bitpos % 8))) == 0) return false;
    h += delta;
  }
  return true;
}

uint32_t MergeableBloom::BloomHash(const Slice& key) {
  return Hash(key.data(), key.size(), 0xbc9f1d34);
}

}//namespace leveldb
