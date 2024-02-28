//
// Created by yyw on 2024/1/25.
//

#ifndef GDH_MERGEABLEBLOOM_H
#define GDH_MERGEABLEBLOOM_H

#include <string>
#include <vector>

#include "../include/leveldb/options.h"
#include "../include/leveldb/slice.h"

namespace leveldb {


class MergeableBloom {
 public:
  explicit MergeableBloom(const Options& options);

  MergeableBloom(const MergeableBloom&) = delete;
  MergeableBloom& operator=(const MergeableBloom) = delete;

  //~MergeableBloom();

  void AddKey(Slice &key);
  void Merge(MergeableBloom* bloom);
  const char* GetResult();
  bool KeyMayMatch(Slice& key);

 private:
  void GenerateFilter();
  void CreateFilter(const Slice* key,int n);
  uint32_t BloomHash(const Slice& key);

  size_t bits_per_key;
  size_t k_;
  std::string key;
  std::vector<size_t> start_;
  std::vector<size_t> tmp_keys;
  char* result;
  size_t reslut_size;
};

}



#endif  // GDH_MERGEABLEBLOOM_H
