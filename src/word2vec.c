//  Copyright 2013 Google Inc. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <arpa/inet.h>
#include <stdint.h>
#include <errno.h>
#include <malloc.h>

#define MAX_STRING_ARRAY 10
#define MAX_STRING 100
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40

const int vocab_hash_size = 200*1000*1000;  // Maximum 200 * 0.7 = 140M words in the vocabulary

#ifdef __GNUC__
long long memory_allocated = 0;
void free_( void* ptr)
{
	memory_allocated -= malloc_usable_size( ptr);
	free( ptr);
}
void* malloc_( int size) {
	void* ptr = malloc( size);
	if (ptr) memory_allocated += malloc_usable_size( ptr);
	return ptr;
}
void* calloc_( int memb, int size) {
	void* ptr = calloc( memb, size);
	if (ptr) memory_allocated += malloc_usable_size( ptr);
	return ptr;
}
void* realloc_( void* ptr, int size) {
	int prevsize = ptr ? malloc_usable_size( ptr) : 0;
	ptr = realloc( ptr, size);
	if (ptr) memory_allocated += malloc_usable_size( ptr) - prevsize;
	return ptr;
}
#else
#define free_ free
#define malloc_ malloc
#define calloc_ calloc
#define realloc_ realloc
#endif

typedef struct Dictionary
{
	char* mem;
	size_t memidx;
	size_t memsize;
} Dictionary;

void initDictionary( Dictionary* dict)
{
	dict->memsize = 1<<20;
	dict->mem = malloc( dict->memsize);
	if (dict->mem == NULL)
	{
		fprintf( stderr, "out of memory\n");
		exit(1);
	}
	dict->memidx = 0;
}

void swapDictionary( Dictionary* dict1, Dictionary* dict2)
{
	Dictionary tmp;
	memcpy( &tmp, dict1, sizeof(Dictionary));
	memcpy( dict1, dict2, sizeof(Dictionary));
	memcpy( dict2, &tmp, sizeof(Dictionary));
}

size_t allocDictionaryHandle( Dictionary* dict, const char* word, size_t size)
{
	if (dict->memidx + size + 1 > dict->memsize)
	{
		size_t newsize = dict->memsize * 2;
		while (dict->memidx + size + 1 > newsize)
		{
			newsize *= 2;
		}
		char* newmem = realloc( dict->mem, newsize);
		if (newsize < dict->memsize || newmem == NULL)
		{
			fprintf( stderr, "out of memory\n");
			exit(1);
		}
		dict->mem = newmem;
		dict->memsize = newsize;
	}
	size_t rt = dict->memidx;
	memcpy( dict->mem + dict->memidx, word, size);
	dict->mem[ dict->memidx + size] = 0;
	dict->memidx += size + 1;
	return rt;
}
char* getDictionaryString( Dictionary* dict, long long hnd)
{
	return dict->mem + hnd;
}
void compactDictionary( Dictionary* dict)
{
	char* newmem = realloc( dict->mem, dict->memidx);
	if (newmem) dict->mem = newmem;
}
void freeDictionary( Dictionary* dict)
{
	free( dict->mem);
	memset( dict, 0, sizeof( *dict));
}
static Dictionary dictionary;


#undef USE_DOUBLE_PRECISION_FLOAT
#ifdef USE_DOUBLE_PRECISION_FLOAT
typedef double real;                   // Precision of float numbers
typedef uint64_t real_net_t;
#else
typedef float real;                    // Precision of float numbers
typedef uint32_t real_net_t;
#endif

struct vocab_word {
  long long cn;
  int *point;
  size_t wordhnd;
  char *code, codelen;
};

char train_file[MAX_STRING], output_file[MAX_STRING];
char save_vocab_file[MAX_STRING], read_vocab_file[MAX_STRING];
char word_prefix_use_always_buf[ MAX_STRING] = "";
char* word_prefix_use_always[ MAX_STRING_ARRAY] = {0};
struct vocab_word *vocab;
int binary = 0, portable = 0, cbow = 0, debug_mode = 2, window = 5, min_count = 5, num_threads = 1, min_reduce = 1;
int *vocab_hash;
long long vocab_max_size = 1000, vocab_size = 0, layer1_size = 100;
long long train_words = 0, word_count_actual = 0, file_size = 0, classes = 0;
real alpha = 0.025, starting_alpha, sample = 0;
real *syn0, *syn1, *syn1neg, *expTable;
clock_t start;

int hs = 1, negative = 0;
const int table_size = 1e8;
int *table;

#ifdef USE_DOUBLE_PRECISION_FLOAT
uint64_t htonlf( double a) {
  union
  {
    uint64_t iv;
    uint32_t hv[2];
    double fv;
  } value;
  value.fv = a;
  uint64_t iv = value.iv;
  value.hv[0] = htonl( iv >> 32);
  value.hv[1] = htonl( iv & 0xFFffFFff);
  return value.iv;
}

double ntohlf( uint64_t a) {
  union
  {
    uint64_t iv;
    uint32_t hv[2];
    double fv;
  } value;
  value.iv = a;
  value.hv[0] = ntohl( value.hv[0]);
  value.hv[1] = ntohl( value.hv[1]);
  uint64_t iv = value.hv[0];
  iv <<= 32;
  iv |= value.hv[1];
  value.iv = iv;
  return value.fv;
}
real ntohr( real_net_t a) {
  return ntohlf(a);
}
real_net_t htonr( real a) {
  return htonlf(a);
}
#else
uint32_t htonf( float a) {
  union
  {
    uint32_t iv;
    float fv;
  } value;
  value.fv = a;
  value.iv = htonl( value.iv);
  return value.iv;
}

float ntohf( uint32_t a) {
  union
  {
    uint32_t iv;
    float fv;
  } value;
  value.iv = ntohl( a);
  return value.fv;
}
real ntohr( real_net_t a) {
  return ntohf(a);
}
real_net_t htonr( real a) {
  return htonf(a);
}
#endif 

// Hash function 'sdbm' from 'http://www.cse.yorku.ca/~oz/hash.html'
unsigned long sdbm_hash( const char *str)
{
  unsigned char const* si = (unsigned char const*)str;
  unsigned long hash = 5381;
  unsigned char c = *si++;
  for (; c; c = *si++) {hash = c + (hash << 6) + (hash << 16) - hash;}
  return hash;
}

void InitAlwaysUsedWords()
{
  int i = 0, bi = 0, last_i = 0;
  for (i = 0; i < MAX_STRING && word_prefix_use_always_buf[i]; ++i) {
    if (word_prefix_use_always_buf[i] == ',') {
      word_prefix_use_always_buf[i] = '\0';
      if (i == last_i) {
        last_i = i+1;
        continue;
      }
      word_prefix_use_always[ bi++] = word_prefix_use_always_buf + last_i;
      last_i = i+1;

      if (bi >= MAX_STRING_ARRAY-1) {
        fprintf(stderr, "array passed with option --always exceeds size %d\n", (int)MAX_STRING_ARRAY);
        exit(1);
      }
    }
  }
  if (word_prefix_use_always_buf[last_i]) {
     word_prefix_use_always[ bi++] = word_prefix_use_always_buf + last_i;
  }
  word_prefix_use_always[ bi] = 0;

  char** ai = word_prefix_use_always;
  for (; *ai; ++ai) {
    fprintf( stderr, "do always use words with prefix '%s'\n", *ai);
  }
}

static int doUseAlways( char const* word)
{
  char** ai = word_prefix_use_always;
  for (; *ai; ++ai) {
    int pi = 0;
    for (; (*ai)[pi] && word[pi] && (*ai)[pi] == word[pi]; ++pi){}
    if (!(*ai)[pi]) {
      return 1;
    }
  }
  return 0;
}


void InitUnigramTable() {
  int a, i;
  long long train_words_pow = 0;
  real d1, power = 0.75;
  table = (int *)malloc_(table_size * sizeof(int));
  if (table == NULL) {
    fprintf(stderr, "cannot allocate memory for the table\n");
    exit(1);
  }
  for (a = 0; a < vocab_size; a++) train_words_pow += pow(vocab[a].cn, power);
  i = 0;
  d1 = pow(vocab[i].cn, power) / (real)train_words_pow;
  for (a = 0; a < table_size; a++) {
    table[a] = i;
    if (a / (real)table_size > d1) {
      i++;
      d1 += pow(vocab[i].cn, power) / (real)train_words_pow;
    }
    if (i >= vocab_size) i = vocab_size - 1;
  }
}

// Reads a single word from a file, assuming space + tab + EOL to be word boundaries
void ReadWord(char *word, FILE *fin) {
  int a = 0, ch;
  while (!feof(fin)) {
    ch = fgetc(fin);
    if (ch == 13) continue;
    if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
      if (a > 0) {
        if (ch == '\n') ungetc(ch, fin);
        break;
      }
      if (ch == '\n') {
        strcpy(word, (char *)"</s>");
        return;
      } else continue;
    }
    word[a] = ch;
    a++;
    if (a >= MAX_STRING - 1) a--;   // Truncate too long words
  }
  word[a] = 0;
}

// Returns hash value of a word
int GetWordHash( const char *word) {
  return sdbm_hash( word) % vocab_hash_size;
}

// Returns position of a word in the vocabulary; if the word is not found, returns -1
int SearchVocab(char *word) {
  unsigned int hash = GetWordHash(word);
  while (1) {
    if (vocab_hash[hash] == -1) return -1;
    if (!strcmp(word, getDictionaryString( &dictionary, vocab[vocab_hash[hash]].wordhnd))) return vocab_hash[hash];
    hash = (hash + 1) % vocab_hash_size;
  }
  return -1;
}

// Reads a word and returns its index in the vocabulary
int ReadWordIndex(FILE *fin) {
  char word[MAX_STRING];
  ReadWord(word, fin);
  if (feof(fin)) return -1;
  return SearchVocab(word);
}

// Adds a word to the vocabulary
int AddWordToVocab(char *word) {
  unsigned int hash, length = strlen(word);
  if (length >= MAX_STRING) length = MAX_STRING-1;
  vocab[vocab_size].wordhnd = allocDictionaryHandle( &dictionary, word, length);
  vocab[vocab_size].cn = 0;
  vocab_size++;
  // Reallocate memory if needed
  if (vocab_size + 2 >= vocab_max_size) {
    vocab_max_size += 1000;
    vocab = (struct vocab_word *)realloc_(vocab, vocab_max_size * sizeof(struct vocab_word));
    if (vocab == NULL)
    {
        fprintf(stderr, "out of memory\n");
        exit(1);
    }
  }
  hash = GetWordHash(word);
  while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
  vocab_hash[hash] = vocab_size - 1;
  return vocab_size - 1;
}

// Used later for sorting by word counts
int VocabCompare(const void *a, const void *b) {
    return ((struct vocab_word *)b)->cn - ((struct vocab_word *)a)->cn;
}

void DestroyVocab() {
  int a;

  for (a = 0; a < vocab_size; a++) {
    if (vocab[a].code != NULL) {
      free(vocab[a].code);
    }
    if (vocab[a].point != NULL) {
      free(vocab[a].point);
    }
  }
  free(vocab);
}

// Sorts the vocabulary by frequency using word counts
void SortVocab() {
  int a, size, new_a;
  unsigned int hash;
  // Sort the vocabulary and keep </s> at the first position
  qsort(&vocab[1], vocab_size - 1, sizeof(struct vocab_word), VocabCompare);
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  size = vocab_size;
  train_words = 0;
  Dictionary new_dictionary;
  initDictionary( &new_dictionary);

  for (new_a = 1, a = 1; a < size; a++) { // Skip </s>
    // Words with a prefix marked as '--always' will always be used
    const char* word = getDictionaryString( &dictionary, vocab[a].wordhnd);
    int useAlways = doUseAlways( word);

    // Words occuring less than min_count times will be discarded from the vocab
    if (vocab[a].cn >= min_count || useAlways) {
      vocab[ new_a].cn = vocab[ a].cn;
      vocab[ new_a].wordhnd = allocDictionaryHandle( &new_dictionary, word, strlen(word));
      // Hash will be re-computed, as after the sorting it is not actual
      hash=GetWordHash( word);
      while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
      vocab_hash[ hash] = new_a;
      train_words += vocab[ new_a].cn;
      ++new_a;
    }
  }
  vocab_size = new_a;
  vocab = (struct vocab_word *)realloc_( vocab, (vocab_size + 1) * sizeof(struct vocab_word));
  if (vocab == NULL)
  {
    fprintf(stderr, "out of memory\n");
    exit(1);
  }
  memset( vocab + vocab_size, 0, sizeof( struct vocab_word));
  // ... no clue what this additional element allocated (realloc above) is used for. we are just nulling it, just in case

  swapDictionary( &dictionary, &new_dictionary);
  freeDictionary( &new_dictionary);

  // Allocate memory for the binary tree construction
  for (a = 0; a < vocab_size; a++) {
    vocab[a].code = (char *)calloc_(MAX_CODE_LENGTH, sizeof(char));
    vocab[a].point = (int *)calloc_(MAX_CODE_LENGTH, sizeof(int));
    if (vocab[a].code == NULL || vocab[a].point == NULL)
    {
      fprintf(stderr, "out of memory\n");
      exit(1);
    }
  }
}

// Create binary Huffman tree using the word counts
// Frequent words will have short uniqe binary codes
void CreateBinaryTree() {
  long long a, b, i, min1i, min2i, pos1, pos2, point[MAX_CODE_LENGTH];
  char code[MAX_CODE_LENGTH];
  long long *count = (long long *)calloc_(vocab_size * 2 + 1, sizeof(long long));
  long long *binary = (long long *)calloc_(vocab_size * 2 + 1, sizeof(long long));
  long long *parent_node = (long long *)calloc_(vocab_size * 2 + 1, sizeof(long long));
  if (count == NULL || binary == NULL || parent_node == NULL)
  {
    fprintf(stderr, "out of memory\n");
    exit(1);
  }
  for (a = 0; a < vocab_size; a++) count[a] = vocab[a].cn;
  for (a = vocab_size; a < vocab_size * 2; a++) count[a] = 1e15;
  pos1 = vocab_size - 1;
  pos2 = vocab_size;
  // Following algorithm constructs the Huffman tree by adding one node at a time
  for (a = 0; a < vocab_size - 1; a++) {
    // First, find two smallest nodes 'min1, min2'
    if (pos1 >= 0) {
      if (count[pos1] < count[pos2]) {
        min1i = pos1;
        pos1--;
      } else {
        min1i = pos2;
        pos2++;
      }
    } else {
      min1i = pos2;
      pos2++;
    }
    if (pos1 >= 0) {
      if (count[pos1] < count[pos2]) {
        min2i = pos1;
        pos1--;
      } else {
        min2i = pos2;
        pos2++;
      }
    } else {
      min2i = pos2;
      pos2++;
    }
    count[vocab_size + a] = count[min1i] + count[min2i];
    parent_node[min1i] = vocab_size + a;
    parent_node[min2i] = vocab_size + a;
    binary[min2i] = 1;
  }
  // Now assign binary code to each vocabulary word
  for (a = 0; a < vocab_size; a++) {
    b = a;
    i = 0;
    while (1) {
      code[i] = binary[b];
      point[i] = b;
      i++;
      b = parent_node[b];
      if (b == vocab_size * 2 - 2) break;
    }
    vocab[a].codelen = i;
    vocab[a].point[0] = vocab_size - 2;
    for (b = 0; b < i; b++) {
      vocab[a].code[i - b - 1] = code[b];
      vocab[a].point[i - b] = point[b] - vocab_size;
    }
  }
  free(count);
  free(binary);
  free(parent_node);
}

void LearnVocabFromTrainFile() {
  char word[MAX_STRING];
  FILE *fin;
  long long a, i;
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: cannot open training data file: %s\n", strerror(errno));
    exit(1);
  }
  vocab_size = 0;
  AddWordToVocab((char *)"</s>");
  while (1) {
    ReadWord(word, fin);
    if (feof(fin)) break;
    train_words++;
    if ((debug_mode > 1) && (train_words % 1000000 == 0)) {
#ifdef __GNUC__
      printf("words %d million, vocab size=%u million, memory %u mega bytes\n", (unsigned int)(train_words / 1000000), (unsigned int)(vocab_size / 1000000), (unsigned int)(memory_allocated >> 20));
#else
      printf("words %d million, vocab size=%u million\n", (unsigned int)(train_words / 1000000), (unsigned int)(vocab_size / 1000000));
#endif
      fflush(stdout);
    }
    i = SearchVocab(word);
    if (i == -1) {
      a = AddWordToVocab(word);
      vocab[a].cn = 1;
    } else vocab[i].cn++;
    if (vocab_size > vocab_hash_size * 0.7) {
      fprintf( stderr, "hash table size too small");
      exit(1);
    }
  }
  SortVocab();
  compactDictionary( &dictionary);
  if (debug_mode > 0) {
    printf("Vocab size: %lld\n", vocab_size);
    printf("Words in train file: %lld\n", train_words);
#ifdef __GNUC__
    printf("Memory allocated for vocabulary: %u mega bytes\n", (unsigned int)(memory_allocated >> 20));
#endif
  }
  file_size = ftell(fin);
  fclose(fin);
}

void SaveVocab() {
  long long i;
  FILE *fo = fopen(save_vocab_file, "wb");
  if (fo == NULL) {
     printf("ERROR: cannot open vocab file: %s\n", strerror(errno));
  } else {
    for (i = 0; i < vocab_size; i++) fprintf(fo, "%s %lld\n", getDictionaryString( &dictionary, vocab[i].wordhnd), vocab[i].cn);
    fclose(fo);
  }
}

void ReadVocab() {
  long long a, i = 0;
  char c;
  char word[MAX_STRING];
  FILE *fin = fopen(read_vocab_file, "rb");
  if (fin == NULL) {
    printf("Cannot open vocabulary file %s\n", strerror(errno));
    exit(1);
  }
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  vocab_size = 0;
  while (1) {
    ReadWord(word, fin);
    if (feof(fin)) break;
    a = AddWordToVocab(word);
    fscanf(fin, "%lld%c", &vocab[a].cn, &c);
    i++;
  }
  SortVocab();
  compactDictionary( &dictionary);
  if (debug_mode > 0) {
    printf("Vocab size: %lld\n", vocab_size);
    printf("Words in train file: %lld\n", train_words);
#ifdef __GNUC__
    printf("Memory allocated for vocabulary: %u mega bytes\n", (unsigned int)(memory_allocated >> 20));
#endif
  }
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: Cannot open training data file: %s\n", strerror(errno));
    exit(1);
  }
  fseek(fin, 0, SEEK_END);
  file_size = ftell(fin);
  fclose(fin);  
}

void InitNet() {
  long long a, b;
  a = posix_memalign((void **)&syn0, 128, (long long)vocab_size * layer1_size * sizeof(real));
  if (syn0 == NULL) {printf("Memory allocation failed\n"); exit(1);}
  if (hs) {
    a = posix_memalign((void **)&syn1, 128, (long long)vocab_size * layer1_size * sizeof(real));
    if (syn1 == NULL) {printf("Memory allocation failed\n"); exit(1);}
    for (b = 0; b < layer1_size; b++) for (a = 0; a < vocab_size; a++)
     syn1[a * layer1_size + b] = 0;
  }
  if (negative>0) {
    a = posix_memalign((void **)&syn1neg, 128, (long long)vocab_size * layer1_size * sizeof(real));
    if (syn1neg == NULL) {printf("Memory allocation failed\n"); exit(1);}
    for (b = 0; b < layer1_size; b++) for (a = 0; a < vocab_size; a++)
     syn1neg[a * layer1_size + b] = 0;
  }
  for (b = 0; b < layer1_size; b++) for (a = 0; a < vocab_size; a++)
   syn0[a * layer1_size + b] = (rand() / (real)RAND_MAX - 0.5) / layer1_size;
  CreateBinaryTree();
}

void DestroyNet() {
  if (syn0 != NULL) {
    free(syn0);
  }
  if (syn1 != NULL) {
    free(syn1);
  }
  if (syn1neg != NULL) {
    free(syn1neg);
  }
}

void *TrainModelThread(void *id) {
  long long a, b, d, word, last_word, sentence_length = 0, sentence_position = 0;
  long long word_count = 0, last_word_count = 0, sen[MAX_SENTENCE_LENGTH + 1];
  long long l1, l2, c, target, label;
  unsigned long long next_random = (long long)id;
  real f, g;
  clock_t now;
  real *neu1 = (real *)calloc_(layer1_size, sizeof(real));
  real *neu1e = (real *)calloc_(layer1_size, sizeof(real));
  if (neu1 == NULL || neu1e == NULL)
  {
    fprintf( stderr, "out of memory\n");
    exit(1);
  }
  FILE *fi = fopen(train_file, "rb");
  if (fi == NULL) {
    fprintf(stderr, "no such file or directory: %s", train_file);
    exit(1);
  }
  fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
  while (1) {
    if (word_count - last_word_count > 10000) {
      word_count_actual += word_count - last_word_count;
      last_word_count = word_count;
      if ((debug_mode > 1)) {
        now=clock();
        printf("%cAlpha: %f  Progress: %.2f%%  Words/thread/sec: %.2fk  ", 13, alpha,
         word_count_actual / (real)(train_words + 1) * 100,
         word_count_actual / ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000));
        fflush(stdout);
      }
      alpha = starting_alpha * (1 - word_count_actual / (real)(train_words + 1));
      if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
    }
    if (sentence_length == 0) {
      while (1) {
        word = ReadWordIndex(fi);
        if (feof(fi)) break;
        if (word == -1) continue;
        word_count++;
        if (word == 0) break;
        // The subsampling randomly discards frequent words while keeping the ranking same
        if (sample > 0) {
          real ran = (sqrt(vocab[word].cn / (sample * train_words)) + 1) * (sample * train_words) / vocab[word].cn;
          next_random = next_random * (unsigned long long)25214903917 + 11;
          if (ran < (next_random & 0xFFFF) / (real)65536) continue;
        }
        sen[sentence_length] = word;
        sentence_length++;
        if (sentence_length >= MAX_SENTENCE_LENGTH) break;
      }
      sentence_position = 0;
    }
    if (feof(fi)) break;
    if (word_count > train_words / num_threads) break;
    word = sen[sentence_position];
    if (word == -1) continue;
    for (c = 0; c < layer1_size; c++) neu1[c] = 0;
    for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
    next_random = next_random * (unsigned long long)25214903917 + 11;
    b = next_random % window;
    if (cbow) {  //train the cbow architecture
      // in -> hidden
      for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
        c = sentence_position - window + a;
        if (c < 0) continue;
        if (c >= sentence_length) continue;
        last_word = sen[c];
        if (last_word == -1) continue;
        for (c = 0; c < layer1_size; c++) neu1[c] += syn0[c + last_word * layer1_size];
      }
      if (hs) for (d = 0; d < vocab[word].codelen; d++) {
        f = 0;
        l2 = vocab[word].point[d] * layer1_size;
        // Propagate hidden -> output
        for (c = 0; c < layer1_size; c++) f += neu1[c] * syn1[c + l2];
        if (f <= -MAX_EXP) continue;
        else if (f >= MAX_EXP) continue;
        else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
        // 'g' is the gradient multiplied by the learning rate
        g = (1 - vocab[word].code[d] - f) * alpha;
        // Propagate errors output -> hidden
        for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1[c + l2];
        // Learn weights hidden -> output
        for (c = 0; c < layer1_size; c++) syn1[c + l2] += g * neu1[c];
      }
      // NEGATIVE SAMPLING
      if (negative > 0) for (d = 0; d < negative + 1; d++) {
        if (d == 0) {
          target = word;
          label = 1;
        } else {
          next_random = next_random * (unsigned long long)25214903917 + 11;
          target = table[(next_random >> 16) % table_size];
          if (target == 0) target = next_random % (vocab_size - 1) + 1;
          if (target == word) continue;
          label = 0;
        }
        l2 = target * layer1_size;
        f = 0;
        for (c = 0; c < layer1_size; c++) f += neu1[c] * syn1neg[c + l2];
        if (f > MAX_EXP) g = (label - 1) * alpha;
        else if (f < -MAX_EXP) g = (label - 0) * alpha;
        else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
        for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg[c + l2];
        for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * neu1[c];
      }
      // hidden -> in
      for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
        c = sentence_position - window + a;
        if (c < 0) continue;
        if (c >= sentence_length) continue;
        last_word = sen[c];
        if (last_word == -1) continue;
        for (c = 0; c < layer1_size; c++) syn0[c + last_word * layer1_size] += neu1e[c];
      }
    } else {  //train skip-gram
      for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
        c = sentence_position - window + a;
        if (c < 0) continue;
        if (c >= sentence_length) continue;
        last_word = sen[c];
        if (last_word == -1) continue;
        l1 = last_word * layer1_size;
        for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
        // HIERARCHICAL SOFTMAX
        if (hs) for (d = 0; d < vocab[word].codelen; d++) {
          f = 0;
          l2 = vocab[word].point[d] * layer1_size;
          // Propagate hidden -> output
          for (c = 0; c < layer1_size; c++) f += syn0[c + l1] * syn1[c + l2];
          if (f <= -MAX_EXP) continue;
          else if (f >= MAX_EXP) continue;
          else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
          // 'g' is the gradient multiplied by the learning rate
          g = (1 - vocab[word].code[d] - f) * alpha;
          // Propagate errors output -> hidden
          for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1[c + l2];
          // Learn weights hidden -> output
          for (c = 0; c < layer1_size; c++) syn1[c + l2] += g * syn0[c + l1];
        }
        // NEGATIVE SAMPLING
        if (negative > 0) for (d = 0; d < negative + 1; d++) {
          if (d == 0) {
            target = word;
            label = 1;
          } else {
            next_random = next_random * (unsigned long long)25214903917 + 11;
            target = table[(next_random >> 16) % table_size];
            if (target == 0) target = next_random % (vocab_size - 1) + 1;
            if (target == word) continue;
            label = 0;
          }
          l2 = target * layer1_size;
          f = 0;
          for (c = 0; c < layer1_size; c++) f += syn0[c + l1] * syn1neg[c + l2];
          if (f > MAX_EXP) g = (label - 1) * alpha;
          else if (f < -MAX_EXP) g = (label - 0) * alpha;
          else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
          for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg[c + l2];
          for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * syn0[c + l1];
        }
        // Learn weights input -> hidden
        for (c = 0; c < layer1_size; c++) syn0[c + l1] += neu1e[c];
      }
    }
    sentence_position++;
    if (sentence_position >= sentence_length) {
      sentence_length = 0;
      continue;
    }
  }
  fclose(fi);
  free(neu1);
  free(neu1e);
  pthread_exit(NULL);
}

void TrainModel() {
  long a, b, c, d;
  FILE *fo;
  pthread_t *pt = (pthread_t *)malloc_(num_threads * sizeof(pthread_t));
  if (pt == NULL) {
    fprintf(stderr, "cannot allocate memory for threads\n");
    exit(1);
  }
  printf("Starting training using file %s\n", train_file);
  starting_alpha = alpha;
  if (read_vocab_file[0] != 0) ReadVocab(); else LearnVocabFromTrainFile();
  if (save_vocab_file[0] != 0) SaveVocab();
  if (output_file[0] == 0) return;
  InitNet();
  if (negative > 0) InitUnigramTable();
  start = clock();
  for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainModelThread, (void *)a);
  for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
  fo = fopen(output_file, "wb");
  if (fo == NULL) {
    fprintf(stderr, "Cannot open %s: %s\n", output_file, strerror(errno));
    exit(1);
  }
  if (classes == 0) {
    // Save the word vectors
    fprintf(fo, "%lld %lld\n", vocab_size, layer1_size);
    for (a = 0; a < vocab_size; a++) {
      fprintf(fo, "%s ", getDictionaryString( &dictionary, vocab[a].wordhnd));
      if (binary) {
        if (portable) {
          for (b = 0; b < layer1_size; b++) {
            real_net_t v_n = htonr(syn0[a * layer1_size + b]);
            fwrite(&v_n, sizeof(real_net_t), 1, fo);
          }
        } else {
          for (b = 0; b < layer1_size; b++) {
            fwrite(&syn0[a * layer1_size + b], sizeof(real), 1, fo);
          }
        }
      }
      else for (b = 0; b < layer1_size; b++) {
        fprintf(fo, "%lf ", syn0[a * layer1_size + b]);
      }
      fprintf(fo, "\n");
    }
  } else {
    // Run K-means on the word vectors
    int clcn = classes, iter = 10, closeid;
    int *centcn = (int *)malloc_(classes * sizeof(int));
    if (centcn == NULL) {
      fprintf(stderr, "cannot allocate memory for centcn\n");
      exit(1);
    }
    int *cl = (int *)calloc_(vocab_size, sizeof(int));
    if (cl == NULL)
    {
      fprintf(stderr, "out of memory\n");
      exit(1);
    }
    real closev, x;
    real *cent = (real *)calloc_(classes * layer1_size, sizeof(real));
    if (cent == NULL)
    {
      fprintf(stderr, "out of memory\n");
      exit(1);
    }
    for (a = 0; a < vocab_size; a++) cl[a] = a % clcn;
    for (a = 0; a < iter; a++) {
      for (b = 0; b < clcn * layer1_size; b++) cent[b] = 0;
      for (b = 0; b < clcn; b++) centcn[b] = 1;
      for (c = 0; c < vocab_size; c++) {
        for (d = 0; d < layer1_size; d++) {
          cent[layer1_size * cl[c] + d] += syn0[c * layer1_size + d];
          centcn[cl[c]]++;
        }
      }
      for (b = 0; b < clcn; b++) {
        closev = 0;
        for (c = 0; c < layer1_size; c++) {
          cent[layer1_size * b + c] /= centcn[b];
          closev += cent[layer1_size * b + c] * cent[layer1_size * b + c];
        }
        closev = sqrt(closev);
        for (c = 0; c < layer1_size; c++) cent[layer1_size * b + c] /= closev;
      }
      for (c = 0; c < vocab_size; c++) {
        closev = -10;
        closeid = 0;
        for (d = 0; d < clcn; d++) {
          x = 0;
          for (b = 0; b < layer1_size; b++) x += cent[layer1_size * d + b] * syn0[c * layer1_size + b];
          if (x > closev) {
            closev = x;
            closeid = d;
          }
        }
        cl[c] = closeid;
      }
    }
    // Save the K-means classes
    for (a = 0; a < vocab_size; a++) fprintf(fo, "%s %d\n", getDictionaryString( &dictionary, vocab[a].wordhnd), cl[a]);
    free(centcn);
    free(cent);
    free(cl);
  }
  fclose(fo);
  free(table);
  free(pt);
  DestroyVocab();
}

int ArgPos(char *str, int argc, char **argv) {
  int a;
  for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
    if (a == argc - 1) {
      printf("Argument missing for %s\n", str);
      exit(1);
    }
    return a;
  }
  return -1;
}

int main(int argc, char **argv) {
  int i;
  if (argc == 1) {
    printf("WORD VECTOR estimation toolkit v 0.1b\n\n");
    printf("Options:\n");
    printf("Parameters for training:\n");
    printf("\t-train <file>\n");
    printf("\t\tUse text data from <file> to train the model\n");
    printf("\t-output <file>\n");
    printf("\t\tUse <file> to save the resulting word vectors / word clusters\n");
    printf("\t-size <int>\n");
    printf("\t\tSet size of word vectors; default is 100\n");
    printf("\t-window <int>\n");
    printf("\t\tSet max skip length between words; default is 5\n");
    printf("\t-sample <float>\n");
    printf("\t\tSet threshold for occurrence of words. Those that appear with higher frequency");
    printf(" in the training data will be randomly down-sampled; default is 0 (off), useful value is 1e-5\n");
    printf("\t-hs <int>\n");
    printf("\t\tUse Hierarchical Softmax; default is 1 (0 = not used)\n");
    printf("\t-negative <int>\n");
    printf("\t\tNumber of negative examples; default is 0, common values are 5 - 10 (0 = not used)\n");
    printf("\t-threads <int>\n");
    printf("\t\tUse <int> threads (default 1)\n");
    printf("\t-min-count <int>\n");
    printf("\t\tThis will discard words that appear less than <int> times; default is 5\n");
    printf("\t-alpha <float>\n");
    printf("\t\tSet the starting learning rate; default is 0.025\n");
    printf("\t-classes <int>\n");
    printf("\t\tOutput word classes rather than word vectors; default number of classes is 0 (vectors are written)\n");
    printf("\t-debug <int>\n");
    printf("\t\tSet the debug mode (default = 2 = more info during training)\n");
    printf("\t-binary <int>\n");
    printf("\t\tSave the resulting vectors in binary moded; default is 0 (off)\n");
    printf("\t-portable <int>\n");
    printf("\t\tIn case of binary, save the resulting binary vectors in portable network byte order; default is 0 (off)\n");
    printf("\t-always <prefix>\n");
    printf("\t\tDo use index words with prefix <prefix> (separated by commas)\n");
    printf("\t-save-vocab <file>\n");
    printf("\t\tThe vocabulary will be saved to <file>\n");
    printf("\t-read-vocab <file>\n");
    printf("\t\tThe vocabulary will be read from <file>, not constructed from the training data\n");
    printf("\t-cbow <int>\n");
    printf("\t\tUse the continuous back of words model; default is 0 (skip-gram model)\n");
    printf("\nExamples:\n");
    printf("./word2vec -train data.txt -output vec.txt -debug 2 -size 200 -window 5 -sample 1e-4 -negative 5 -hs 0 -binary 0 -cbow 1\n\n");
    return 0;
  }
  initDictionary( &dictionary);
  output_file[0] = 0;
  save_vocab_file[0] = 0;
  read_vocab_file[0] = 0;
  if ((i = ArgPos((char *)"-size", argc, argv)) > 0) layer1_size = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-train", argc, argv)) > 0) strcpy(train_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-save-vocab", argc, argv)) > 0) strcpy(save_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-read-vocab", argc, argv)) > 0) strcpy(read_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-debug", argc, argv)) > 0) debug_mode = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-binary", argc, argv)) > 0) binary = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-portable", argc, argv)) > 0) portable = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-cbow", argc, argv)) > 0) cbow = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-output", argc, argv)) > 0) strcpy(output_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-window", argc, argv)) > 0) window = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-sample", argc, argv)) > 0) sample = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-hs", argc, argv)) > 0) hs = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-min-count", argc, argv)) > 0) min_count = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-classes", argc, argv)) > 0) classes = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-always", argc, argv)) > 0) strcpy(word_prefix_use_always_buf, argv[i + 1]);

  InitAlwaysUsedWords();

  vocab = (struct vocab_word *)calloc_(vocab_max_size, sizeof(struct vocab_word));
  vocab_hash = (int *)calloc_(vocab_hash_size, sizeof(int));
  expTable = (real *)malloc_((EXP_TABLE_SIZE + 1) * sizeof(real));
  if (vocab == NULL || vocab_hash == NULL || expTable == NULL) {
    fprintf(stderr, "out of memory\n");
    exit(1);
  }
  for (i = 0; i < EXP_TABLE_SIZE; i++) {
    expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
    expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
  }
  TrainModel();
  DestroyNet();
  free(vocab_hash);
  free(expTable);
  freeDictionary( &dictionary);
  return 0;
}
