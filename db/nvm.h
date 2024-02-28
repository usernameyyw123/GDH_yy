//
// Created by yyw on 2024/2/27.
//
#ifndef GDH_NVM_H
#define GDH_NVM_H

#include <stdlib.h>
#include <iostream>
#include "numa.h"
#include "../include/leveldb/options.h"


 namespace leveldb {

 extern int nvm_node;
 extern int nvm_next_node;
 extern size_t nvm_free_space;
 extern bool nvm_node_has_changed;

  void NvmNodeSizeInit(const Options& options_);
  void NvmNodeSizeRecord(size_t s);
  void numa_node_size(int node, long long *freep);
  //
} //namespace leveldb

#endif  // GDH_NVM_H
