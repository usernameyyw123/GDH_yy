// Copyright (c) 2011 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

#ifndef STORAGE_LEVELDB_DB_WRITE_BATCH_INTERNAL_H_
#define STORAGE_LEVELDB_DB_WRITE_BATCH_INTERNAL_H_

#include "db/dbformat.h"
#include "leveldb/write_batch.h"

// jeffxu
#include <unordered_map>
// jeffxu
//
#include<chrono>
//
namespace leveldb {

class MemTable;
//
struct KeyValue {
  int accessCount;
  std::chrono::steady_clock::time_point  lastAccessTime;
  double  hotValue;

  KeyValue():accessCount(0),hotValue(0.0){
    lastAccessTime = std::chrono::steady_clock::now();
  }
};
class HeatCalcuator{
 public:
  void updateHeat(KeyValue& kv){
    auto current_time = std::chrono::steady_clock::now();
    std::chrono::duration<double>  elapsed_seconds = current_time - kv.lastAccessTime;

    double timeDiff = elapsed_seconds.count();
    double  accessWeight = 1.0;
    double decayRate = 0.05;

    kv.hotValue = (kv.accessCount * accessWeight)/(1.0 + decayRate * timeDiff);
    kv.lastAccessTime = current_time;
  }
  void access(KeyValue& kv){
    kv.accessCount++;
    updateHeat(kv);
  }
};
//
// WriteBatchInternal provides static methods for manipulating a
// WriteBatch that we don't want in the public WriteBatch interface.
class WriteBatchInternal {
 public:
  // Return the number of entries in the batch.
  static uint32_t Count(const WriteBatch* batch);

  // Set the count for the number of entries in the batch.
  static void SetCount(WriteBatch* batch, uint32_t n);

  // Return the sequence number for the start of this batch.
  static SequenceNumber Sequence(const WriteBatch* batch);

  // Store the specified number as the sequence number for the start of
  // this batch.
  static void SetSequence(WriteBatch* batch, SequenceNumber seq);

  static Slice Contents(const WriteBatch* batch) { return Slice(batch->rep_); }

  static size_t ByteSize(const WriteBatch* batch) { return batch->rep_.size(); }

  static void SetContents(WriteBatch* batch, const Slice& contents);

  static Status InsertInto(const WriteBatch* batch, MemTable* memtable);
  static Status InsertInto(const WriteBatch* batch, MemTable* memtable,
                           uint64_t& pos, uint64_t file_numb);
  // jeffxu
  static Status InsertInto(const WriteBatch* batch, MemTable* memtable,
                           std::unordered_map<std::string, int>& memhashtable,
                           MutilMemPolicys& mmpolicy, uint64_t& pos,
                           uint64_t file_numb);
  static Status InsertIntoInPlace(
      const WriteBatch* batch, MemTable* memtable,
      std::unordered_map<std::string, int>& memhashtable,
      MutilMemPolicys& mmpolicy, uint64_t new_log_numb, uint64_t new_log_potion,
      uint64_t new_content_size);
  // jeffxu
//
  static bool ShouldJudgeHotValue(int& count,const int number );
  static Status HotValueMoveToFirst(std::unordered_map<std::string,int>& key_buff_1_data,
                                    std::unordered_map<std::string, int>& key_buff_2_data);

  static Status InsertIntoHotColdBuff(std::unordered_map<std::string,int> key_buff_1,std::unordered_map<std::string,std::string> &hot_data_,
                                      std::unordered_map<std::string,std::string> &cold_data_,std::string key,std::string value);
  static void Clear(int& count);
  // 从batch的pos位置解析出一条kv对，并把pos更新为下一条记录在batch中偏移，isdel代表这条kv记录是不是删除操作
  static Status ParseRecord(const WriteBatch* batch, uint64_t& pos, Slice& key,
                            Slice& value, bool& isDel);
  static void Append(WriteBatch* dst, const WriteBatch* src);
};

}  // namespace leveldb

#endif  // STORAGE_LEVELDB_DB_WRITE_BATCH_INTERNAL_H_
