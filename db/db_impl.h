// Copyright (c) 2011 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

#ifndef STORAGE_LEVELDB_DB_DB_IMPL_H_
#define STORAGE_LEVELDB_DB_DB_IMPL_H_

#include <deque>
#include <set>
#include <unordered_map>
#include "db/dbformat.h"
#include "db/log_writer.h"
#include "db/snapshot.h"
#include "db/vlog_manager.h"
#include "db/vlog_reader.h"
#include "db/vlog_writer.h"
#include "leveldb/db.h"
#include "leveldb/env.h"
#include "port/port.h"
#include "port/thread_annotations.h"

// jeffxu
#include <atomic>
#include <map>
#include <unordered_map>
#include <vector>

#include "cudasrc/kernelwrapper.cuh"
#include "leveldb/filter_policy.h"
#include "util/newbloom.h"
// jeffxu

namespace leveldb {

// jeffxu

// every bloom filter entry max numbers;
static const size_t kPerBloomEntry = 100000;
static const int mempolicy_count_ = 3;

static int hash_entry_count = 0;

 extern std::unordered_map<std::string, uint32_t> hot_data;
 extern std::unordered_map<std::string, uint32_t> cold_data;
//
//static std::unordered_map<std::string,int> key_buffer_1_store;
//static std::unordered_map<std::string,int> key_buffer_2_store;
extern size_t access_count;
static int data_access_count_ = 0;
static int per_access_number = 100000;
//

static int hot_data_count = 50000;
static int cold_data_count = 100000;

static void HandleHotColdData() {
  if (!hot_data.empty()) {
    for (const auto &it : hot_data) {
      if (cold_data.size() < cold_data_count) {
        cold_data[it.first] = it.second;
      } else {
        cudaKernelFunc();
        cold_data.clear();
        break;
      }
    }
  }
  hot_data.clear();
}

// 记录无效空间的结构体
typedef struct RecordSpace {
  uint64_t vlog_number_;
  uint64_t entry_position_;
  uint32_t entry_size_;
  uint32_t entry_hot_;

  RecordSpace()
      : vlog_number_(-1),
        entry_position_(-1),
        entry_size_(-1),
        entry_hot_(-1){};
} RSpace;

static std::atomic_int space_position;

struct MutilMemPolicys {
  BloomFilter *mempolicys_[mempolicy_count_];
  int entrycounts_[mempolicy_count_];


  BloomFilter *mempolicy_F1_;
  BloomFilter *mempolicy_F2_;
  BloomFilter *mempolicy_F3_;
  std::string filterstring_F1_;
  std::string filterstring_F2_;
  std::string filterstring_F3_;
  //
  std::unordered_map<std::string,int> key_buffer_1_store;
  std::unordered_map<std::string,int> key_buffer_2_store;
  //
  int F1_count_;
  int F2_count_;
  int F3_count_;
  int current_count_;

  MutilMemPolicys() : current_count_(0) {
    for (int i = 0; i < mempolicy_count_; i++) {
      mempolicys_[i] = new BloomFilter(10 * kPerBloomEntry, kPerBloomEntry);
      entrycounts_[i] = 0;
    }
    mempolicy_F1_ = new BloomFilter(10 * kPerBloomEntry, kPerBloomEntry);
    mempolicy_F2_ = new BloomFilter(10 * kPerBloomEntry, kPerBloomEntry);
    mempolicy_F3_ = new BloomFilter(10 * kPerBloomEntry, kPerBloomEntry);
    key_buffer_1_store = {};
    key_buffer_2_store = {};

  }
};
// jeffxu

class MemTable;

class TableCache;

class Version;

class VersionEdit;

class VersionSet;

class GarbageCollector;

class DBImpl : public DB {
 public:
  DBImpl(const Options &options, const std::string &dbname);

  virtual ~DBImpl();

  // Implementations of the DB interface
  virtual Status Put(const WriteOptions &, const Slice &key,
                     const Slice &value);

  virtual Status Delete(const WriteOptions &, const Slice &key);

  virtual Status Write(const WriteOptions &options, WriteBatch *updates);

  virtual Status Get(const ReadOptions &options, const Slice &key,
                     std::string *value);

  //  virtual Status GetNoLock(const ReadOptions& options,
  //                   const Slice& key,
  //                 std::string* value);
  virtual Status GetPtr(const ReadOptions &options, const Slice &key,
                        std::string *value);

  virtual Iterator *NewIterator(const ReadOptions &);

  //  virtual const Snapshot* GetSnapshot();//不支持快照
  //  virtual void ReleaseSnapshot(const Snapshot* snapshot);//同上
  virtual bool GetProperty(const Slice &property, std::string *value);

  virtual void GetApproximateSizes(const Range *range, int n, uint64_t *sizes);

  virtual void CompactRange(const Slice *begin, const Slice *end);

  // 因为从sst文件和memtable获得的v只是vlog的索引
  // 需要从vlog文件读出索引位置处的value值,val_ptr是索引，value是存放真正v的
  Status RealValue(Slice val_ptr, std::string *value);

  // jeffxu
  ///////////////////////////////////////
  virtual void PrintMyStats();

  void cudaKernelWrapper();
  ///////////////////////////////////////
  // jeffxu

  // Extra methods (for testing) that are not in the public DB interface

  // Compact any files in the named level that overlap [*begin,*end]
  void TEST_CompactRange(int level, const Slice *begin, const Slice *end);

  // Force current memtable contents to be compacted.
  Status TEST_CompactMemTable();

  // Return an internal iterator over the current state of the database.
  // The keys of this iterator are internal keys (see format.h).
  // The returned iterator should be deleted when no longer needed.
  Iterator *TEST_NewInternalIterator();

  // Return the maximum overlapping data (in bytes) at next level for any
  // file at a level >= 1.
  int64_t TEST_MaxNextLevelOverlappingBytes();

  // Record a sample of bytes read at the specified internal key.
  // Samples are taken approximately once every config::kReadBytesPeriod
  // bytes.
  void RecordReadSample(Slice key);

  void CleanVlog();

  bool has_cleaned_;

  void MaybeScheduleClean(bool isManualClean = false);

  bool IsShutDown() { return shutting_down_.Acquire_Load(); }

  int TotalVlogFiles();

  uint64_t GetVlogNumber();

 private:
  friend class DB;

  friend class GarbageCollector;

  struct CompactionState;
  struct Writer;

  Iterator *NewInternalIterator(const ReadOptions &,
                                SequenceNumber *latest_snapshot,
                                uint32_t *seed);

  Status NewDB();

  // Recover the descriptor from persistent storage.  May do a significant
  // amount of work to recover recently logged updates.  Any changes to
  // be made to the descriptor are added to *edit.
  Status Recover(VersionEdit *edit, bool *save_manifest)
      EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  void MaybeIgnoreError(Status *s) const;

  // Delete any unneeded files and stale in-memory entries.
  void DeleteObsoleteFiles();

  // Compact the in-memory write buffer to disk.  Switches to a new
  // log-file/memtable and writes a new descriptor iff successful.
  // Errors are recorded in bg_error_.
  void CompactMemTable() EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  Status RecoverVlogFile(bool *save_manifest, VersionEdit *edit,
                         SequenceNumber *max_sequence);

  // 数据库恢复是靠vlog文件恢复
  Status RecoverLogFile(uint64_t log_number, bool last_log, bool *save_manifest,
                        VersionEdit *edit, SequenceNumber *max_sequence)
      EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  Status WriteLevel0Table(MemTable *mem, VersionEdit *edit, Version *base)
      EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  Status MakeRoomForWrite(bool force /* compact even if there is room? */)
      EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  WriteBatch *BuildBatchGroup(Writer **last_writer);

  void RecordBackgroundError(const Status &s);

  void MaybeScheduleCompaction() EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  static void BGWork(void *db);

  void BackgroundCall();

  void BackgroundCompaction() EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  static void BGClean(void *db);

  static void BGCleanAll(void *db);

  static void BGCleanRecover(void *db);

  void BackgroundClean();

  void BackgroundCleanAll();

  void BackgroundRecoverClean();

  void CleanupCompaction(CompactionState *compact)
      EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  Status DoCompactionWork(CompactionState *compact);

  EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  Status OpenCompactionOutputFile(CompactionState *compact);

  Status FinishCompactionOutputFile(CompactionState *compact, Iterator *input);

  Status InstallCompactionResults(CompactionState *compact)
      EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  //////////////////////////////////////////////
  Status DoReverseCompactionWork(CompactionState *compact)
      EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  Status InstallReverseCompactionResults(CompactionState *compact)
      EXCLUSIVE_LOCKS_REQUIRED(mutex_);

  // std::map<std::string,struct RecordSpace> spacemap;

  RSpace *space_arr;

  // std::vector<RSpace> spacemap;

  // Constant after construction
  Env *const env_;
  const InternalKeyComparator internal_comparator_;
  const InternalFilterPolicy internal_filter_policy_;
  const Options options_;  // options_.comparator == &internal_comparator_
  bool owns_info_log_;
  bool owns_cache_;
  const std::string dbname_;

  // table_cache_ provides its own synchronization
  TableCache *table_cache_;

  // Lock over the persistent DB state.  Non-NULL iff successfully acquired.
  FileLock *db_lock_;

  // State below is protected by mutex_
  port::Mutex mutex_;
  port::AtomicPointer shutting_down_;
  port::CondVar bg_cv_;  // Signalled when background work finishes
  MemTable *mem_;
  MemTable *imm_;                // Memtable being compacted
  port::AtomicPointer has_imm_;  // So bg thread can detect non-NULL imm_

  // jeffxu
  // hash表和级联bloom-filter
  std::unordered_map<std::string, int> inmemory_keys_buffer_;
  MutilMemPolicys mutilmempolicys_;
  // jeffxu

  uint64_t logfile_number_;  // 当前vlog文件的编号
  log::VWriter *vlog_;       // 写vlog的包装类
  WritableFile *vlogfile_;   // vlog文件写打开
  uint64_t vlog_head_;       // 当前vlog文件的偏移写
  uint64_t check_point_;     // vlog文件的重启点
  uint64_t check_log_;       // 从那个vlog文件开始回放
  uint64_t
      drop_count_;  // 合并产生了多少条垃圾记录，这些新产生的信息还没有持久化到sst文件
  uint64_t recover_clean_vlog_number_;
  uint64_t recover_clean_pos_;
  VlogManager vlog_manager_;
  uint32_t seed_;  // For sampling.

  // Queue of writers.
  std::deque<Writer *> writers_;
  WriteBatch *tmp_batch_;

  //  SnapshotList snapshots_;

  // Set of table files to protect from deletion because they are
  // part of ongoing compactions.
  std::set<uint64_t> pending_outputs_;

  // Has a background compaction been scheduled or is running?
  bool bg_compaction_scheduled_;
  bool bg_clean_scheduled_;
  // Information for a manual compaction
  struct ManualCompaction {
    int level;
    bool done;
    const InternalKey *begin;  // NULL means beginning of key range
    const InternalKey *end;    // NULL means end of key range
    InternalKey tmp_storage;   // Used to keep track of compaction progress
  };
  ManualCompaction *manual_compaction_;

  VersionSet *versions_;

  // Have we encountered a background error in paranoid mode?
  Status bg_error_;

  // Per level compaction stats.  stats_[level] stores the stats for
  // compactions that produced data for the specified "level".
  struct CompactionStats {
    int64_t micros;
    int64_t bytes_read;
    int64_t bytes_written;

    CompactionStats() : micros(0), bytes_read(0), bytes_written(0) {}

    void Add(const CompactionStats &c) {
      this->micros += c.micros;
      this->bytes_read += c.bytes_read;
      this->bytes_written += c.bytes_written;
    }
  };

  CompactionStats stats_[config::kNumLevels];

  // No copying allowed
  DBImpl(const DBImpl &);

  void operator=(const DBImpl &);

  const Comparator *user_comparator() const {
    return internal_comparator_.user_comparator();
  }
};

// Sanitize db options.  The caller should delete result.info_log if
// it is not equal to src.info_log.
extern Options SanitizeOptions(const std::string &db,
                               const InternalKeyComparator *icmp,
                               const InternalFilterPolicy *ipolicy,
                               const Options &src);

}  // namespace leveldb

#endif  // STORAGE_LEVELDB_DB_DB_IMPL_H_
