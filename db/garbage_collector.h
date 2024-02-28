
#ifndef STORAGE_LEVELDB_DB_GARBAGE_COLLECTOR_H_
#define STORAGE_LEVELDB_DB_GARBAGE_COLLECTOR_H_

#include "stdint.h"
#include "db/vlog_reader.h"

namespace leveldb{
class VReader;
class DBImpl;
class VersionEdit;

class GarbageCollector
{
    public:
        GarbageCollector(DBImpl* db):vlog_number_(0), garbage_pos_(0), vlog_reader_(NULL), db_(db){}
        ~GarbageCollector(){delete vlog_reader_;}
        void SetVlog(uint64_t vlog_number, uint64_t garbage_beg_pos=0);
        ///////////////////////////
        // pre GarbageCollect, collect the invalid space
        void PreGarbageCollect(VersionEdit* edit, bool* save_edit);
        uint64_t GetGarbagerFileNumber();
        ///////////////////////////
        void BeginGarbageCollect(VersionEdit* edit, bool* save_edit);

    private:
        uint64_t vlog_number_;
        uint64_t garbage_pos_;//vlog文件起始垃圾回收的地方
        log::VReader* vlog_reader_;
        DBImpl* db_;
        /////////////////
        int Kgroupnumber_ = 4;
        int Kgroupsize_ = 4 * 1024 *1024;
        /////////////////
};

}

#endif
