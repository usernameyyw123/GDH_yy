// Copyright (c) 2011 The LevelDB Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file. See the AUTHORS file for names of contributors.

#include "db/vlog_writer.h"

#include <stdint.h>

#include "leveldb/env.h"
#include "util/coding.h"
#include "util/crc32c.h"

namespace leveldb {
namespace log {

VWriter::VWriter(WritableFile* dest) : dest_(dest) {}

VWriter::~VWriter() {}

// 记录格式为
// CRC32固定4位，记录长度编码成变长64位的记录，后面是writebatch的全部内容
Status VWriter::AddRecord(const Slice& slice, int& head_size) {
  const char* ptr = slice.data();
  size_t left = slice.size();
  // 存放 checksum 和 编码后的 value size，分别是4B和最大10B
  char buf[kVHeaderMaxSize];
  uint32_t crc = crc32c::Extend(0, ptr, left);
  crc = crc32c::Mask(crc);  // Adjust for storage
  EncodeFixed32(buf, crc);
  char* end = EncodeVarint64(&buf[4], left);
  assert(end != nullptr);
  head_size = 4 + (end - &buf[4]);
  Status s = dest_->Append(Slice(buf, head_size));
  if (s.ok()) {
    s = dest_->Append(Slice(ptr, left));  // 写一条物理记录就刷一次
    if (s.ok()) {
      s = dest_->Flush();
    }
  }
  return s;
}

Status VWriter::InPlaceAddRecord(const Slice& slice, int& head_size,
                                 uint32_t vlog_num) {
  const char* ptr = slice.data();
  size_t left = slice.size();
  char buf[kVHeaderMaxSize];
  uint32_t crc = crc32c::Extend(0, ptr, left);
  crc = crc32c::Mask(crc);  // Adjust for storage
  EncodeFixed32(buf, crc);
  char* end = EncodeVarint64(&buf[4], left);
  assert(end != NULL);
  head_size = 4 + (end - &buf[4]);
  Status s = dest_->Append(Slice(buf, head_size));
  if (s.ok()) {
    s = dest_->Append(Slice(ptr, left));  // 写一条物理记录就刷一次
    if (s.ok()) {
      s = dest_->Flush();
    }
  }
  return s;
}

}  // namespace log
}  // namespace leveldb
