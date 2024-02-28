// Copyright (c) 2011 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.
//
// WriteBatch::rep_ :=
//    sequence: fixed64
//    count: fixed32
//    data: record[count]
// record :=
//    kTypeValue varstring varstring         |
//    kTypeDeletion varstring
// varstring :=
//    len: varint32
//    data: uint8[len]

#include "leveldb/write_batch.h"

#include "db/dbformat.h"
#include "db/memtable.h"
#include "db/write_batch_internal.h"
#include "leveldb/db.h"
#include "util/coding.h"
#include "db/db_impl.h"
// jeffxu
#include <iostream>
#include <unordered_map>
#include <chrono>
#include "cudasrc/kernelwrapper.cuh"
// jeffxu

namespace leveldb {

// jeffxu

extern int hash_entry_count;

// extern std::vector<std::string> memkeyhot;
// extern std::vector<std::string> memkeycold;

extern std::unordered_map<std::string, uint32_t> hot_data;
extern std::unordered_map<std::string, uint32_t > cold_data;
//
//extern std::unordered_map<std::string,int> key_buffer_1_store;
//extern std::unordered_map<std::string,int> key_buffer_2_store;
size_t access_count = 1;
extern int per_access_number;
extern int data_access_count_ ;
const int hot_access_count_ = 5;
//
extern int hot_data_count;
extern int cold_data_count;

// jeffxu
extern void HandleHotColdData();
// WriteBatch header has an 8-byte sequence number followed by a 4-byte count.
static const size_t kHeader = 12;

WriteBatch::WriteBatch() { Clear(); }

WriteBatch::~WriteBatch() {}

WriteBatch::Handler::~Handler() {}

void WriteBatch::Clear() {
  rep_.clear();
  rep_.resize(kHeader);
}

size_t WriteBatch::ApproximateSize() { return rep_.size(); }

Status WriteBatch::ParseRecord(uint64_t &pos, Slice &key, Slice &value,
                               bool &isDel) const {
  Slice input(rep_);
  input.remove_prefix(pos);

  const char *begin_pos = input.data();
  char tag = input[0];
  input.remove_prefix(1);
  switch (tag) {
    case kTypeValue: {
      if (!(GetLengthPrefixedSlice(&input, &key) &&
            GetLengthPrefixedSlice(&input, &value))) {
        return Status::Corruption("bad WriteBatch Put");
      }
      isDel = false;
      break;
    }
    case kTypeDeletion: {
      if (!GetLengthPrefixedSlice(&input, &key)) {
        return Status::Corruption("bad WriteBatch Delete");
      }
      isDel = true;
      break;
    }
    default:
      return Status::Corruption("unknown WriteBatch tag");
  }
  pos += (input.data() - begin_pos);
  return Status::OK();
}

Status WriteBatch::Iterate(Handler *handler) const {
  Slice input(rep_);
  if (input.size() < kHeader) {
    return Status::Corruption("malformed WriteBatch (too small)");
  }

  input.remove_prefix(kHeader);
  Slice key, value;
  int found = 0;
  while (!input.empty()) {
    found++;
    data_access_count_++;
    char tag = input[0];
    input.remove_prefix(1);
    switch (tag) {
      case kTypeValue:
        if (GetLengthPrefixedSlice(&input, &key) &&
            GetLengthPrefixedSlice(&input, &value)) {
          handler->Put(key, value);
        } else {
          return Status::Corruption("bad WriteBatch Put");
        }
        break;
      case kTypeDeletion:
        if (GetLengthPrefixedSlice(&input, &key)) {
          handler->Delete(key);
        } else {
          return Status::Corruption("bad WriteBatch Delete");
        }
        break;
      default:
        return Status::Corruption("unknown WriteBatch tag");
    }
  }
  if (found != WriteBatchInternal::Count(this)) {
    return Status::Corruption("WriteBatch has wrong count");
  } else {
    return Status::OK();
  }
}

Status WriteBatch::Iterate(Handler *handler, uint64_t &pos, uint64_t file_numb)
    const {  // pos是当前vlog文件的大小
  Slice input(rep_);
  if (input.size() < kHeader) {
    return Status::Corruption("malformed WriteBatch (too small)");
  }
  const char *last_pos = input.data();
  input.remove_prefix(
      kHeader);  // 移除掉WriteBatch的头部，它的头部前8个字节表示队首kv对的sequence
  // 后4字节代表WriteBatch包含了多少个kv对。
  pos +=
      kHeader;  // 因为vlog记录的是WriteBatch，所以这kHeader字节也会被写入vlog
  last_pos += kHeader;  // last_pos就是记录上一条记录插入vlog后vlog文件的大小
  Slice key, value;
  int found = 0;
  while (!input.empty()) {  // 遍历WriteBatch的每一条kv对
    found++;
    char tag = input[0];
    input.remove_prefix(1);  // 判断kv类型
    switch (tag) {
      case kTypeValue:
        if (GetLengthPrefixedSlice(&input, &key) &&
            GetLengthPrefixedSlice(&input, &value)) {
          const char *now_pos = input.data();  // 如果是插入，解析出k和v
          size_t len = now_pos - last_pos;     // 计算出这条记录的大小
          last_pos = now_pos;

          std::string v;
          PutVarint64(&v, len);
          PutVarint32(&v, file_numb);
          PutVarint64(&v, pos);
          handler->Put(key, v);
          pos = pos + len;  // 更新pos
        } else {
          return Status::Corruption("bad WriteBatch Put");
        }
        break;
      case kTypeDeletion:
        if (GetLengthPrefixedSlice(&input, &key)) {
          const char *now_pos = input.data();
          size_t len = now_pos - last_pos;
          pos = pos + len;  // 对于删除操作，不需要v值，更新pos
          last_pos = now_pos;

          handler->Delete(key);  // delete的val是不是要写成文件号？
        } else {
          return Status::Corruption("bad WriteBatch Delete");
        }
        break;
      default:
        return Status::Corruption("unknown WriteBatch tag");
    }
  }
  if (found != WriteBatchInternal::Count(this)) {
    return Status::Corruption("WriteBatch has wrong count");
  } else {
    return Status::OK();
  }
}

// jeffxu
// param list :
// handler无效，后续会删除，mem hashtable就是key的缓存，
// pos是当前vlog文件的大小，file number是文件号
// WriteBatch 格式：
// sequence number (7 Bytes)| Value Type (1 Byte) | KV pair count (4 Byte)
// KV对格式: Type (1 Byte) | K len (var) | key (fix) | V len  (var) | V (fix)
Status WriteBatch::Iterate(Handler *handler,
                           std::unordered_map<std::string, int> &memhashtable,
                           MutilMemPolicys &mmpolicy, uint64_t &pos,
                           uint64_t file_number) const {
  Slice input(rep_);
  if (input.size() < kHeader) {
    return Status::Corruption("malformed WriteBatch (too small)");
  }
  const char *last_pos = input.data();
  // 移除掉WriteBatch的头部，它的头部前8个字节表示队首kv对的sequence
  // 后4字节代表WriteBatch包含了多少个kv对。
  input.remove_prefix(kHeader);
  // 因为vlog记录的是WriteBatch，所以这kHeader字节也会被写入vlog
  pos += kHeader;
  last_pos += kHeader;  // last_pos就是记录上一条记录插入vlog后vlog文件的大小
  Slice key, value;
  int found = 0;
  while (!input.empty()) {  // 遍历WriteBatch的每一条kv对
    found++;
    data_access_count_++;
    char tag = input[0];
    input.remove_prefix(1);  // 判断kv类型
    switch (tag) {
      case kTypeValue:
        if (GetLengthPrefixedSlice(&input, &key) &&
            GetLengthPrefixedSlice(&input, &value)) {
          //std::string KVoffeset;
          //
          std::string key_data;
          bool key_found_in_buffer_1 = false;
          bool key_found_in_buffer_2 = false;
          //
          uint32_t hot_value = 0;
          //KVoffeset.append(key.data(), key.size());
          //
          key_data.append(key.data(),key.size());
          //
          // *************************
          //  判断冷热的逻辑
          // *************************
           //如果key不在bloom-current里面，且count没有超过kPerBloomEntry，则热度为0，并且将key加入bloom-current
        //  int first_index = (0 + mmpolicy.current_count_) % 3;
        //  int second_index = (1 + mmpolicy.current_count_) % 3;
        //  int third_index = (2 + mmpolicy.current_count_) % 3;
        //  // 如果需要重置current
       //   if (mmpolicy.entrycounts_[first_index] >= kPerBloomEntry) {
      //      mmpolicy.mempolicys_[first_index]->reset();
      //      mmpolicy.current_count_ = (++mmpolicy.current_count_) % 3;
      //      mmpolicy.entrycounts_[first_index] = 0;
      //    } else {  // 如果不需要重置current
      //      // 如果filter1里面不存在这个key
      //      if (!mmpolicy.mempolicys_[first_index]->VaryExist(
      //            KVoffeset.c_str())) {
      //        mmpolicy.mempolicys_[first_index]->SetKey(KVoffeset.c_str());
      //        mmpolicy.entrycounts_[first_index]++;
       //       hot_value = 0 ;
      //      } else {  // 如果filter1里面存在这个key, 到filter2里面去查
      //        if (!mmpolicy.mempolicys_[second_index]->VaryExist(
       //               KVoffeset.c_str())) {
       //         mmpolicy.mempolicys_[second_index]->SetKey(KVoffeset.c_str());
       //         mmpolicy.entrycounts_[second_index]++;
       //         hot_value = 0 + 2;
       //       } else {  // 如果filter2里面存在这个key,到filter3里面去查
       //         if (!mmpolicy.mempolicys_[third_index]->VaryExist(
       //                 KVoffeset.c_str())) {
        //          mmpolicy.mempolicys_[third_index]->SetKey(KVoffeset.c_str());
       //           mmpolicy.entrycounts_[third_index]++;
        //          hot_value = 0 + 2 + 4;
        //        } else {
         //         // 设置一个最大值，说明这个热数据很热（三个bloom
        //          // filter依然不能鉴别其热度）
        //          hot_value = 1 << 7;
        //        }
        //      }
        //    }
        //  }
          //
          //std::chrono::steady_clock::time_point startAccessTime ;
          //startAccessTime = std::chrono::steady_clock::now();
          auto it_1 = mmpolicy.key_buffer_1_store.find(key_data);
          if(it_1 != mmpolicy.key_buffer_1_store.end()) {
            key_found_in_buffer_1 = true;
            hot_value = ++(it_1->second);
          } else {
            auto it_2 = mmpolicy.key_buffer_2_store.find(key_data);
            if(it_2 != mmpolicy.key_buffer_2_store.end()) {
              key_found_in_buffer_2 = true;
              it_2->second++;
              hot_value = 1;
            } else if(!key_found_in_buffer_1 && !key_found_in_buffer_2) {
              mmpolicy.key_buffer_1_store[key_data] = access_count;
              hot_value = access_count;

            }
          }
          //auto current_time = std::chrono::steady_clock::now();
          //std::chrono::duration<double> elapsed_second = current_time - startAccessTime;
          //printf("%f\n",elapsed_second);
          bool b = WriteBatchInternal::ShouldJudgeHotValue(data_access_count_,per_access_number);
          if(b) {
            Status s = WriteBatchInternal::HotValueMoveToFirst(
                mmpolicy.key_buffer_1_store,mmpolicy.key_buffer_2_store);
            if(!s.ok()) {
              Status::Corruption("move to first error");
            } else{
               WriteBatchInternal::Clear(data_access_count_);
            }
          }
          //
          //*************************
          // 判断冷热的逻辑结束
          //*************************

          const char *now_pos = input.data();  // 如果是插入，解析出k和v
          size_t len = now_pos - last_pos;     // 计算出这条记录的大小
          last_pos = now_pos;
          std::string v;

          PutVarint64(&v, len);
          PutVarint32(&v, file_number);
          PutVarint64(&v, pos);
          PutVarint32(&v, hot_value);
          handler->Put(key, v);
          pos = pos + len;  // 更新pos

          uint64_t sequence_type;
          SequenceNumber s;
          sequence_type = ((s << 8) | tag);
          char *buffer = new char[sizeof(uint64_t)];
          memcpy(buffer, &sequence_type, sizeof(uint64_t));
          //KVoffeset.append(buffer, sizeof(uint64_t));
          //KVoffeset.append(v.data(), v.size());
          //key_data.append(buffer,sizeof(uint64_t));
          // std::cout << KVoffeset.data() << "__" << std::endl;
          //
          auto it = mmpolicy.key_buffer_1_store.find(key_data);
          key_data.append(buffer,sizeof(uint64_t));
          key_data.append(v.data(),v.size());
          if(it != mmpolicy.key_buffer_1_store.end()) {
            hot_data.insert(std::make_pair(key_data,hot_value));
          } else {
            cold_data.insert(std::make_pair(key_data,hot_value));
          }
          //
          // 测试代码
          //std::string str(key.data(), 16);
          //hot_data.insert(std::make_pair(str, KVoffeset));
          //hot_data.insert(std::make_pair(key_data,v));
          if (hot_data.size() >= hot_data_count) {
            HandleHotColdData();
            // cudaKernelFunc();
            // cudaKernelFunc(memkeyhot);
            // HandleHotColdData();
            // memkeyhot.clear();
          }
          // memhashtable.insert(std::make_pair(KVoffeset,hot_value));
          //KVoffeset.clear();
          key_data.clear();
          delete[] buffer;
          //hash_entry_count++;
          //buffer = nullptr;
        } else {
          return Status::Corruption("bad WriteBatch Put");
        }
        break;
      case kTypeDeletion:
        if (GetLengthPrefixedSlice(&input, &key)) {
          const char *now_pos = input.data();
          size_t len = now_pos - last_pos;
          pos = pos + len;  // 对于删除操作，不需要v值，更新pos
          last_pos = now_pos;
          handler->Delete(key);
        } else {
          return Status::Corruption("bad WriteBatch Delete");
        }
        break;
      default:
        return Status::Corruption("unknown WriteBatch tag");
    }
  }
  if (found != WriteBatchInternal::Count(this)) {
    return Status::Corruption("WriteBatch has wrong count");
  } else {
    return Status::OK();
  }
}

// 原地插入的逻辑，此时只需要将内容写入LSM-Tree中即可，write
// batch丢弃，因为之前已经原地更新到某个log中了
Status WriteBatch::Iterate(Handler *handler,
                           std::unordered_map<std::string, int> &memhashtable,
                           MutilMemPolicys &mmpolicy, uint64_t new_log_numb,
                           uint64_t new_log_position,
                           uint64_t new_content_size) const {
  Slice input(rep_);
  if (input.size() < kHeader) {
    return Status::Corruption("malformed WriteBatch (too small)");
  }
  const char *last_pos = input.data();
  // 移除掉WriteBatch的头部，它的头部前8个字节表示队首kv对的sequence
  input.remove_prefix(kHeader);
  // 后4字节代表WriteBatch包含了多少个kv对。
  // pos += kHeader;
  // 因为vlog记录的是WriteBatch，所以这kHeader字节也会被写入vlog
  last_pos += kHeader;  // last_pos就是记录上一条记录插入vlog后vlog文件的大小
  Slice key, value;
  int found = 0;
  while (!input.empty()) {  // 遍历WriteBatch的每一条kv对
    found++;
    data_access_count_++;
    char tag = input[0];
    input.remove_prefix(1);  // 判断kv类型
    switch (tag) {
      case kTypeValue:
        if (GetLengthPrefixedSlice(&input, &key) &&
            GetLengthPrefixedSlice(&input, &value)) {
          //std::string KVoffeset;
          std::string key_data;
          bool key_found_in_buffer_1 = false;
          bool key_found_in_buffer_2 = false;
          uint32_t hot_value = 0;
        // KVoffeset.append(key.data(), key.size());
          key_data.append(key.data(),key.size());
          //*************************
          // 判断冷热的逻辑
          //*************************
          //如果key不在bloom-current里面，且count没有超过kPerBloomEntry，则热度为0，并且将key加入bloom-current
      //    int first_index = (0 + mmpolicy.current_count_) % 3;
      //    int second_index = (1 + mmpolicy.current_count_) % 3;
       //   int third_index = (2 + mmpolicy.current_count_) % 3;
      //    // 如果需要重置current
      //    if (mmpolicy.entrycounts_[first_index] >= kPerBloomEntry) {

        //    mmpolicy.mempolicys_[first_index]->reset();
       //     mmpolicy.current_count_ = (++mmpolicy.current_count_) % 3;
       //     mmpolicy.entrycounts_[first_index] = 0;
       //   } else {  // 如果不需要重置current
            // 如果filter1里面不存在这个key
       //     if (!mmpolicy.mempolicys_[first_index]->VaryExist(
       //             KVoffeset.c_str())) {
       //       mmpolicy.mempolicys_[first_index]->SetKey(KVoffeset.c_str());
       //       mmpolicy.entrycounts_[first_index]++;
        //      hot_value = 0 ;
      //      } else {  // 如果filter1里面存在这个key, 到filter2里面去查
       //       if (!mmpolicy.mempolicys_[second_index]->VaryExist(
        //              KVoffeset.c_str())) {
        //        mmpolicy.mempolicys_[second_index]->SetKey(KVoffeset.c_str());
        //        mmpolicy.entrycounts_[second_index]++;
       //         hot_value = 0 + 2;
        //      } else {  // 如果filter2里面存在这个key,到filter3里面去查
        //        if (!mmpolicy.mempolicys_[third_index]->VaryExist(
        //                KVoffeset.c_str())) {
        //          mmpolicy.mempolicys_[third_index]->SetKey(KVoffeset.c_str());
         //         mmpolicy.entrycounts_[third_index]++;
          //        hot_value = 0 + 2 + 4;
        //        } else {
                  // 设置一个最大值，说明这个热数据很热（三个bloom
                  // filter依然不能鉴别其热度）
          //        hot_value = 1 << 7;
          //      }
          //    }
           // }
          //}
          //
          auto it_1 = mmpolicy.key_buffer_1_store.find(key_data);
          if(it_1 != mmpolicy.key_buffer_1_store.end()) {
            key_found_in_buffer_1 = true;
            hot_value = ++(it_1->second);
          } else {
            auto it_2 = mmpolicy.key_buffer_2_store.find(key_data);
            if(it_2 != mmpolicy.key_buffer_2_store.end()) {
               key_found_in_buffer_2 = true;
               it_2->second++;
               hot_value = 1;
            } else if(!key_found_in_buffer_1 && !key_found_in_buffer_2) {
               //policy.mempolicys_[first_index]->SetKey(key_data.c_str());
               //policy.mempolicys_[first_index]++;
               mmpolicy.key_buffer_1_store[key_data] = access_count;
               hot_value = access_count;

            }
          }
          //auto current_time = std::chrono::steady_clock::now();
          //std::chrono::duration<double> elapsed_second = current_time - startAccessTime;
          //printf("%f\n",elapsed_second);
          bool b = WriteBatchInternal::ShouldJudgeHotValue(data_access_count_,per_access_number);
          if(b) {
            Status s = WriteBatchInternal::HotValueMoveToFirst(
                mmpolicy.key_buffer_1_store,mmpolicy.key_buffer_2_store);
            if(!s.ok()) {
               Status::Corruption("move to first error");
            } else{
               WriteBatchInternal::Clear(data_access_count_);
            }
          }
          //*************************
          // 判断冷热的逻辑结束
          //*************************

          const char *now_pos = input.data();  // 如果是插入，解析出k和v
          size_t len = now_pos - last_pos;     // 计算出这条记录的大小
          last_pos = now_pos;
          std::string v;

          PutVarint64(&v, new_content_size);
          PutVarint32(&v, new_log_numb);
          PutVarint64(&v, new_log_position);
          PutVarint32(&v, hot_value);
          handler->Put(key, v);
          // pos = pos + len; // 更新pos

          uint64_t sequence_type;
          SequenceNumber s;
          sequence_type = (s << 8 | tag);
          char *buffer = new char[sizeof(uint64_t)];
          memcpy(buffer, &sequence_type, sizeof(uint64_t));
          //KVoffeset.append(buffer, sizeof(uint64_t));
          //KVoffeset.append(v.data(), v.size());
          //key_data.append(buffer, sizeof(uint64_t));
          // std::cout << KVoffeset.data() << "__" << std::endl;
          //
          auto it = mmpolicy.key_buffer_1_store.find(key_data);
          key_data.append(buffer,sizeof(uint64_t));
          key_data.append(v.data(),v.size());
          if(it != mmpolicy.key_buffer_1_store.end()) {
            hot_data.insert(std::make_pair(key_data,hot_value));
          } else {
            cold_data.insert(std::make_pair(key_data,hot_value));
          }
          //
          // 测试代码
          //std::string str(key.data(), 16);
          //hot_data.insert(std::make_pair(str, KVoffeset));
          //hot_data.insert(std::make_pair(key_data,v));
          if (hot_data.size() >= hot_data_count) {
            HandleHotColdData();
            // cudaKernelFunc();
            // cudaKernelFunc(memkeyhot);
            // HandleHotColdData();
            // memkeyhot.clear();
          }
          // memhashtable.insert(std::make_pair(KVoffeset,hotvalue));
          //KVoffeset.clear();
          key_data.clear();
          delete[] buffer;
          //hash_entry_count++;
          //buffer = nullptr;
        } else {
          return Status::Corruption("bad WriteBatch Put");
        }
        break;
      case kTypeDeletion:
        if (GetLengthPrefixedSlice(&input, &key)) {
          const char *now_pos = input.data();
          size_t len = now_pos - last_pos;
          // pos = pos + len;//对于删除操作，不需要v值，更新pos
          last_pos = now_pos;
          handler->Delete(key);
        } else {
          return Status::Corruption("bad WriteBatch Delete");
        }
        break;
      default:
        return Status::Corruption("unknown WriteBatch tag");
    }
  }
  if (found != WriteBatchInternal::Count(this)) {
    return Status::Corruption("WriteBatch has wrong count");
  } else {
    return Status::OK();
  }
}

// jeffxu

uint32_t WriteBatchInternal::Count(const WriteBatch *b) {
  return DecodeFixed32(b->rep_.data() + 8);
}

void WriteBatchInternal::SetCount(WriteBatch *b, uint32_t n) {
  EncodeFixed32(&b->rep_[8], n);
}

SequenceNumber WriteBatchInternal::Sequence(const WriteBatch *b) {
  return SequenceNumber(DecodeFixed64(b->rep_.data()));
}

void WriteBatchInternal::SetSequence(WriteBatch *b, SequenceNumber seq) {
  EncodeFixed64(&b->rep_[0], seq);
}

void WriteBatch::Put(const Slice &key, const Slice &value) {
  // SetCount 从第8个字节开始往rep_中写数据, 前8B 用于存放sequence
  WriteBatchInternal::SetCount(this, WriteBatchInternal::Count(this) + 1);
  rep_.push_back(static_cast<char>(kTypeValue));
  PutLengthPrefixedSlice(&rep_, key);
  PutLengthPrefixedSlice(&rep_, value);
}

void WriteBatch::Delete(const Slice &key) {
  WriteBatchInternal::SetCount(this, WriteBatchInternal::Count(this) + 1);
  rep_.push_back(static_cast<char>(kTypeDeletion));
  PutLengthPrefixedSlice(&rep_, key);
}

namespace {
class MemTableInserter : public WriteBatch::Handler {
 public:
  SequenceNumber sequence_;
  MemTable *mem_;

  virtual void Put(const Slice &key, const Slice &value) {
    mem_->Add(sequence_, kTypeValue, key, value);
    sequence_++;
  }

  virtual void Delete(const Slice &key) {
    mem_->Add(sequence_, kTypeDeletion, key, Slice());
    sequence_++;
  }
};
}  // namespace

Status WriteBatchInternal::InsertInto(const WriteBatch *b, MemTable *memtable) {
  MemTableInserter inserter;
  inserter.sequence_ = WriteBatchInternal::Sequence(b);
  inserter.mem_ = memtable;
  return b->Iterate(&inserter);
}

Status WriteBatchInternal::InsertInto(const WriteBatch *b, MemTable *memtable,
                                      uint64_t &pos, uint64_t file_numb) {
  MemTableInserter inserter;
  inserter.sequence_ = WriteBatchInternal::Sequence(b);
  inserter.mem_ = memtable;
  return b->Iterate(&inserter, pos, file_numb);
}

// jeffxu
Status WriteBatchInternal::InsertInto(
    const WriteBatch *b, MemTable *memtable,
    std::unordered_map<std::string, int> &memhashtable,
    MutilMemPolicys &mmpolicy, uint64_t &pos,
    uint64_t file_numb) {  // pos是当前vlog文件的大小
  MemTableInserter inserter;
  inserter.sequence_ = WriteBatchInternal::Sequence(b);
  inserter.mem_ = memtable;
  return b->Iterate(&inserter, memhashtable, mmpolicy, pos, file_numb);
}

Status WriteBatchInternal::InsertIntoInPlace(
    const WriteBatch *b, MemTable *memtable,
    std::unordered_map<std::string, int> &memhashtable,
    MutilMemPolicys &mmpolicy, uint64_t new_log_numb, uint64_t new_log_potion,
    uint64_t new_content_size) {
  MemTableInserter inserter;
  inserter.sequence_ = WriteBatchInternal::Sequence(b);
  inserter.mem_ = memtable;
  return b->Iterate(&inserter, memhashtable, mmpolicy, new_log_numb,
                    new_log_potion, new_content_size);
}
// jeffxu

void WriteBatchInternal::SetContents(WriteBatch *b, const Slice &contents) {
  assert(contents.size() >= kHeader);
  b->rep_.assign(contents.data(), contents.size());
}

void WriteBatchInternal::Append(WriteBatch *dst, const WriteBatch *src) {
  SetCount(dst, Count(dst) + Count(src));
  assert(src->rep_.size() >= kHeader);
  dst->rep_.append(src->rep_.data() + kHeader, src->rep_.size() - kHeader);
}

Status WriteBatchInternal::ParseRecord(const WriteBatch *batch, uint64_t &pos,
                                       Slice &key, Slice &value, bool &isDel) {
  if (pos < kHeader) pos = kHeader;
  return batch->ParseRecord(pos, key, value, isDel);
}
//
bool WriteBatchInternal::ShouldJudgeHotValue(int& count, const int number) {
  if(count >= number ){
    count -= number ;
    return true;
  }
  return false;

}
Status WriteBatchInternal::HotValueMoveToFirst(std::unordered_map<std::string, int> &key_buff_data_1,
                                               std::unordered_map<std::string, int> &key_buff_data_2) {
  for(auto it = key_buff_data_2.begin(); it != key_buff_data_2.end();){
    it->second--;
    if(it->second >= hot_access_count_){
      key_buff_data_1.insert(std::make_pair(it->first,it->second));
      it = key_buff_data_2.erase(it);
    } else {
      if(it->second < -1){
        it = key_buff_data_2.erase(it);
      } else {
        ++it;
      }
    }
  }
  for(auto it = key_buff_data_1.begin(); it != key_buff_data_1.end();){
    it->second--;
    if(it->second< hot_access_count_ && it->second >= 0){
       key_buff_data_2.insert(std::make_pair(it->first,it->second));
        it = key_buff_data_1.erase(it);
    } else {
        if(it->second < -1){
          it = key_buff_data_1.erase(it);
        } else {
          ++it;
        }
    }
  }
  return Status::OK();
}
Status WriteBatchInternal::InsertIntoHotColdBuff(std::unordered_map<std::string, int> key_data_buff_1,std::unordered_map<std::string,std::string> &hot_data_,
                                                 std::unordered_map<std::string, std::string> &cold_data_,std::string key,std::string value) {
  std::string str(key.data(),key.size());
  auto it = key_data_buff_1.find(key);
  if(it != key_data_buff_1.end()){
    hot_data_.insert(std::make_pair(str,value));
  } else {
    cold_data_.insert(std::make_pair(str,value));
  }
  return Status::OK();
}
void WriteBatchInternal::Clear(int& count) {
  count = 0;
}
}  // namespace leveldb
