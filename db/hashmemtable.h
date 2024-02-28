#ifndef STORAGE_LEVELDB_DB_HASHMEMETABLE_H_
#define STORAGE_LEVELDB_DB_HASHMEMETABLE_H_

#include <string>
#include "../include/leveldb/db.h"
#include "dbformat.h"
#include <assert.h>
#include "../include/leveldb/slice.h"
#include "../include/leveldb/status.h"
#include "skiplist.h"
namespace leveldb{

class InternalKeyComparator;
class MemHashTableIterator;

typedef struct hashnode
{
    /* data */
    int uinkeysize_;
    //提前计算unikey的大小
    char data_[]; 
};

class MemHashTable{
    public:
        explicit MemHashTable(const InternalKeyComparator& comparator);

        void Ref() { ++refs_; }

        void Unref() {
            --refs_;
            assert(refs_ >= 0);
            if (refs_ <= 0) {
            delete this;
            }
        }

        size_t ApproximateMemoryUsage();

        void Add(SequenceNumber seq, ValueType type, const Slice& key, const Slice& value);

        bool Get(const LookupKey& key, std::string* value, Status* s);

    private:

        struct KeyComparator {
            const InternalKeyComparator comparator;
            explicit KeyComparator(const InternalKeyComparator& c) : comparator(c) { }
            int operator()(const char* a, const char* b) const;
        };

        ~MemHashTable();  // Private since only Unref() should be used to delete it     

        int refs_;
        KeyComparator comparator_;

        // No copying allowed
        MemHashTable(const MemHashTable&);
        void operator=(const MemHashTable&);
};


}//leveldb namespace
#endif