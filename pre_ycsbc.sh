#!/bin/bash
sudo rm -rf /usr/local/lib/libleveldb.*
sudo rm -rf /usr/local/include/db
sudo rm -rf /usr/local/include/leveldb
sudo rm -rf /usr/local/include/port
sudo rm -rf /usr/local/include/table
sudo rm -rf /usr/local/include/util
sudo cp -r ./out-shared/libleveldb.so* /usr/local/lib
sudo cp -r ./out-static/libleveldb.a /usr/local/lib
sudo cp -r ./db /usr/local/include
sudo cp -r ./include/leveldb /usr/local/include
sudo cp -r ./port /usr/local/include
sudo cp -r ./table /usr/local/include
sudo cp -r ./util /usr/local/include
sudo ldconfig
