//
// Created by yyw on 2024/2/27.
//
#include "nvm.h"
#include "numa.h"

int leveldb::nvm_node = 2;
int leveldb::nvm_next_node = 4;
size_t leveldb::nvm_free_space = 384L*1024*1024*1024;
bool leveldb::nvm_node_has_changed = false;

namespace leveldb {
void NvmNodeSizeInit(const leveldb::Options &options_) {
  nvm_node = options_.nvm_node;
  nvm_next_node = options_.nvm_next_node;
  long long tmp;
  numa_node_size(nvm_node, &tmp);
  nvm_free_space = (size_t)tmp - 16L * 1024 * 1024;
}
void NvmNodeSizeRecord(size_t s) {
  if (nvm_node_has_changed || nvm_next_node == -1) {
    return;
  }
  size_t tmp = (s + 4095) / (4096*4096);
  if(tmp > nvm_free_space) {
    nvm_node = nvm_next_node;
    nvm_node_has_changed = true;
  } else {
    nvm_free_space -= tmp;
  }
}
void numa_node_size(int node, long long *freep) {
  *freep = node;
}

}// nemapace leveldb
