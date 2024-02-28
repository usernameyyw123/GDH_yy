//
// Created by yyw on 2024/1/25.
//

#ifndef GDH_DATATABLE_H
#define GDH_DATATABLE_H

#include <string>
#include "skiplist.h"
#include "../db/memtable.h"
#include "dbformat.h"
#include "../include/leveldb/options.h"
#include "../include/leveldb/db.h"
#include "../include/leveldb/iterator.h"
#include "../include/leveldb/status.h"
#include "../include/leveldb/slice.h"
#include "../util/mergeablebloom.h"
#include "../util/arena.h"
#include "../include/leveldb/comparator.h"

namespace leveldb {
class IternalKeyComparator;
class DataTableIterator;


class DataTable {

 public:

  explicit DataTable(const IternalKeyComparator& comparator);

  explicit DataTable(const IternalKeyComparator& comparator,MemTable* mem,const Options& options_);

  DataTable(const DataTable&) = delete;

  DataTable& operator=(const DataTable&) = delete;

  ~DataTable();

  void Ref();
  void Unref();

  size_t ApproximateMemoryUsage();

  Iterator* NewIterater();
  //
  bool Get(const LookupKey& key,std::string * value,Status& s);
  void Add(SequenceNumber seq, ValueType type,const Slice& key,const Slice& value);

  Status Compact(DataTable* smalltable,SequenceNumber snum);

 private:

  int refs_;
  KeyComparator comparator_;
  typedef SkipList<const char*, KeyComparator> mTable;

 public:
  MergeableBloom* bloom;
  mTable table_;
  Arena arena_;
  bool IsLastTable;


};

}

#endif  // GDH_DATATABLE_H





