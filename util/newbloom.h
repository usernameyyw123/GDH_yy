#ifndef STORAGE_LEVELDB_UTIL_NEWBLOOM_H_
#define STORAGE_LEVELDB_UTIL_NEWBLOOM_H_

#include <bitset>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

// using namespace std;
namespace leveldb {

typedef unsigned int uint;
#define MAX_BLOOM_N 1 << 20 // 1MB

static int array[] = {5, 7, 11, 13, 31, 37, 61};
class BloomFilter {
 public:
  BloomFilter(int m, int n) : m_m(m), m_n(n) {
    m_k = std::ceil((m / n) * std::log(2));
  }
  virtual ~BloomFilter() {}

  uint RSHash(const char *str, int seed) {
    // unsigned int b = 378551;
    uint a = 63689;
    uint hash = 0;

    while (*str) {
      hash = hash * a + (*str++);
      a *= seed;
    }

    return (hash & 0x7FFFFFFF);
  }

  void SetKey(const char *str) {
    int *p = new int[m_k + 1];
    memset(p, 0, 2);
    for (int i = 0; i < m_k; ++i) {
      p[i] = static_cast<int>(RSHash(str, array[i])) % 1000000;
    }

    for (int j = 0; j < m_k; ++j) {
      bit[p[j]] = true;
    }
    delete[] p;
  }

  int VaryExist(const char *str) {
    int res = 1;
    int *p = new int[m_k + 1];
    memset(p, 0, 2);
    for (int i = 0; i < m_k; ++i) {
      p[i] = static_cast<int>(RSHash(str, array[i])) % 1000000;
    }

    for (int j = 0; j < m_k; ++j) {
      res &= bit[p[j]];
      if (!res) {
        delete[] p;
        return 0;
      }
    }
    delete[] p;
    return 1;
  }

  void reset() { bit.reset(); }

 private:
  // k: number of the hash functions
  // m: the size of bitset
  // n: number of strings to hash (k = [m/n]*ln2)
  int m_k, m_m, m_n;
  std::bitset<MAX_BLOOM_N> bit;
};

}  // namespace leveldb

#endif

// int main(int argc, char *argv[])
// {
// 	BloomFilter bf(5, 2);
//     string str = "hahahehe";
// 	string str2 = "sdfasfa";
// 	bf.SetKey(str.c_str());
// 	int res = bf.VaryExist(str.c_str());

// 	if(res)
// 		cout << "exist" << endl;
// 	else
// 		cout << "not exist" << endl;

// 	//bf.SetKey(str2.c_str());
// 	res = bf.VaryExist(str2.c_str());
// 	if(res)
// 		cout << "exist" << endl;
// 	else
// 		cout << "not exist" << endl;

//     return 0;
// }