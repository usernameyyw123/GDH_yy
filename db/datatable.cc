//
// Created by yyw on 2024/1/25.
//
#include "datatable.h"
#include "dbformat.h"
#include "../util/coding.h"
#include "../include/leveldb/comparator.h"
#include "../include/leveldb/iterator.h"
#include "../include/leveldb/env.h"
#include "skiplist.h"
#include "../util/mergeablebloom.h"
//#include "numa.h"
namespace  leveldb {

static Slice GetLengthPrefixedSlice(const char* data){
  uint32_t len;
  const char* p = data;
  p = GetVarint32Ptr(p,p+5,&len);
  return Slice(p,len);
}

//DataTable::DataTable(const IternalKeyComparator& comparator){}


static const char* EncodeKey(std::string* scratch, const Slice& target) {
  scratch->clear();
  PutVarint32(scratch, target.size());
  scratch->append(target.data(), target.size());
  return scratch->data();
}

class DataTableIterator: public Iterator {
 public:
  explicit DataTableIterator(mTable* table) : iter_(table){}
  DataTableIterator(const DataTableIterator&) = delete;
  DataTableIterator& operator=(const DataTableIterator&) = delete;

  ~DataTableIterator() override = default;

  bool Valid() const override {return iter_.Valid();}
  void Seek(const Slice& target_key) override {
    return iter_.Seek(EncodeKey(&tmps,target_key));
  }
  void SeekToFirst() override {iter_.SeekToFirst();}
  void SeekToLast() override {iter_.SeekToLast();}
  void Next() override {return iter_.Next();}
  void Prev() override {return iter_.Prev();}
  Slice key() const override {return GetLengthPrefixedSlice(iter_.key());}
  Slice value() const override {
    Slice key_slice = GetLengthPrefixedSlice(iter_.key());
    return GetLengthPrefixedSlice(key_slice.data()+key_slice.size());
  }
  Status status() const override {return Status::OK();}
 private:
  mTable::Iterator iter_;
  std::string tmps;
};



Iterator* DataTable::NewIterater() {return new DataTableIterator(&table_);}

bool DataTable::Get(const leveldb::LookupKey& key, std::string* value, leveldb::Status& s) {
  if(bloom != nullptr) {
    Slice tmpkey = key.user_key();
    if(!(bloom->KeyMayMatch(tmpkey))) {
      return false;
    }
  }
  Slice memkey = key.memtable_key();
  mTable ::Iterator iter(&table_);
  iter.Seek(memkey.data());
  if(iter.Valid()) {
    const char* entry = iter.key();
    uint32_t key_length;
    const char* key_ptr = GetVarint32Ptr(entry,entry+5,&key_length);
    if(comparator_.comparator.user_comparator()->Compare
        (Slice(key_ptr,key_length - 8),key.user_key()) == 0) {
      const uint64_t tag = DecodeFixed64(key_ptr + key_length - 8);
      switch (static_cast<ValueType>(tag & 0xff)) {
        case kTypeValue:{
          Slice v = GetLengthPrefixedSlice(key_ptr+key_length);
          value->assign(v.data(),v.size());
          return true;
        }
        case kTypeDeletion:
          Status::NotFound(Slice());
          return true;
      }
    }
  }
  return false;
}

void DataTable::Ref() {++refs_;}
void DataTable::Unref() {
  --refs_;
  assert(refs_ >= 0);
  if(refs_ <= 0) {
    delete this;
  }
}

DataTable::~DataTable(){
  assert(refs_ == 0);
  if(bloom != nullptr) {
    delete bloom;
  }
}

}//namespace levedb