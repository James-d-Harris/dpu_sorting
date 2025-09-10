// host_quicksort.c
#define _POSIX_C_SOURCE 200112L
#include <dpu.h>
#include <dpu_log.h>

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <time.h>
#include <string.h>
#include <assert.h>

// ---------- Build-time knobs (override in Makefile) ----------
#ifndef NR_DPUS
#define NR_DPUS 64
#endif
#ifndef NR_TASKLETS
#define NR_TASKLETS 16
#endif
#ifndef DPU_BINARY
#define DPU_BINARY "./quicksort_dpu"
#endif
#ifndef MAX_ELEMS_PER_DPU
#define MAX_ELEMS_PER_DPU 16384
#endif
#ifdef DPU_ALLOC_PROFILE
#define DPU_ALLOC_PROFILE_STR DPU_ALLOC_PROFILE
#else
#define DPU_ALLOC_PROFILE_STR NULL
#endif

// ---------- Element layout ----------
typedef struct __attribute__((packed, aligned(8))) {
    uint32_t v;
    uint32_t pad;
} elem_t;
_Static_assert(sizeof(elem_t) == 8, "elem_t must be 8 bytes");

// Host <-> DPU args
struct dpu_args { uint32_t n_elems; };

// DPU -> Host stats
struct dpu_stats {
    uint32_t n_elems;
    uint32_t nr_tasklets;
    uint32_t cycles_sort;
    uint32_t cycles_total;
};

// ---------- time helper ----------
static inline uint64_t ns_diff(const struct timespec a, const struct timespec b) {
    return (uint64_t)(b.tv_sec - a.tv_sec) * 1000000000ull
         + (uint64_t)(b.tv_nsec - a.tv_nsec);
}

// ---------- CPU quicksort ----------
static inline void swap_u32(uint32_t *a, uint32_t *b) {
    uint32_t t = *a; *a = *b; *b = t;
}
static inline uint32_t median3_idx(uint32_t *A, size_t i, size_t j, size_t k) {
    uint32_t ai = A[i], aj = A[j], ak = A[k];
    if ((ai < aj) ^ (ai < ak)) return (uint32_t)i;
    if ((aj < ai) ^ (aj < ak)) return (uint32_t)j;
    return (uint32_t)k;
}
static inline void insertion_sort(uint32_t *A, size_t lo, size_t hi) {
    for (size_t i = lo + 1; i <= hi; i++) {
        uint32_t x = A[i];
        size_t j = i;
        while (j > lo && A[j - 1] > x) { A[j] = A[j - 1]; j--; }
        A[j] = x;
    }
}
static void quicksort_u32(uint32_t *A, size_t n) {
    if (n < 2) return;

    // stack cap ~ 2*log2(n) + 8
    size_t t = n, lg = 0; while (t > 1) { t >>= 1; lg++; }
    const size_t STACK_CAP = ((lg << 1) + 8);
    typedef struct { size_t lo, hi; } range_t;
    range_t *stack = (range_t *)malloc(sizeof(range_t) * STACK_CAP);
    size_t sp = 0;

    stack[sp++] = (range_t){0, n - 1};
    const size_t INSERTION_CUTOFF = 32;

    while (sp) {
        range_t rg = stack[--sp];
        size_t lo = rg.lo, hi = rg.hi;
        while (lo < hi) {
            if (hi - lo + 1 <= INSERTION_CUTOFF) { insertion_sort(A, lo, hi); break; }

            size_t mid = lo + ((hi - lo) >> 1);
            size_t piv = median3_idx(A, lo, mid, hi);
            swap_u32(&A[piv], &A[hi]);
            uint32_t pivot = A[hi];

            size_t i = lo;
            for (size_t j = lo; j < hi; j++) {
                if (A[j] <= pivot) { swap_u32(&A[i], &A[j]); i++; }
            }
            swap_u32(&A[i], &A[hi]);

            // Recurse on smaller side first (tail-call elimination)
            size_t left_lo = lo, left_hi = (i == 0) ? 0 : (i - 1);
            size_t right_lo = i + 1, right_hi = hi;

            size_t left_sz  = (left_hi >= left_lo) ? (left_hi - left_lo + 1) : 0;
            size_t right_sz = (right_hi >= right_lo) ? (right_hi - right_lo + 1) : 0;

            if (left_sz < right_sz) {
                if (right_sz > 1 && sp < STACK_CAP) stack[sp++] = (range_t){ right_lo, right_hi };
                hi = (left_sz > 0) ? left_hi : left_lo;
            } else {
                if (left_sz > 1 && sp < STACK_CAP) stack[sp++] = (range_t){ left_lo, left_hi };
                lo = right_lo;
            }
        }
    }
    free(stack);
}

// ---------- k-way merge structs ----------
typedef struct { uint32_t value, shard_id, idx_in_shard; } heap_node_t;
static inline void heap_swap(heap_node_t *h, uint32_t i, uint32_t j) { heap_node_t t = h[i]; h[i] = h[j]; h[j] = t; }
static inline void heap_up(heap_node_t *h, uint32_t i) { while (i>0){uint32_t p=(i-1)>>1; if(h[p].value<=h[i].value)break; heap_swap(h,p,i); i=p;} }
static inline void heap_down(heap_node_t *h, uint32_t n, uint32_t i) {
    for (;;) {
        uint32_t l = 2*i + 1, r = l + 1, m = i;
        if (l < n && h[l].value < h[m].value) m = l;
        if (r < n && h[r].value < h[m].value) m = r;
        if (m == i) break;
        heap_swap(h, i, m); i = m;
    }
}

int main(int argc, char **argv) {
    // ----- Parameters -----
    uint32_t N = (argc >= 2) ? (uint32_t)strtoul(argv[1], NULL, 10) : (1u << 20);
    const uint32_t NB_DPUS = NR_DPUS;
    printf("Sorting N=%u across %u DPUs (NR_TASKLETS=%u)\n", N, NB_DPUS, (unsigned)NR_TASKLETS);

    // ----- Prepare host data -----
    uint32_t *input = (uint32_t *)malloc(sizeof(uint32_t) * N);
    if (!input) { fprintf(stderr, "OOM\n"); return 1; }
    srand(0xC0FFEEu);
    for (uint32_t i = 0; i < N; i++) input[i] = (uint32_t)rand();

    // A copy for CPU-only quicksort (same numbers)
    uint32_t *cpu_arr = (uint32_t *)malloc(sizeof(uint32_t) * N);
    if (!cpu_arr) { fprintf(stderr, "OOM\n"); return 1; }
    memcpy(cpu_arr, input, sizeof(uint32_t) * N);

    // timings
    struct timespec t_after_rand, t_cpu_start, t_cpu_end, t_h2d_start, t_h2d_end, t_launch_end, t_d2h_end, t_merge_end;
    clock_gettime(CLOCK_MONOTONIC, &t_after_rand);

    // ----- CPU quicksort timing -----
    clock_gettime(CLOCK_MONOTONIC, &t_cpu_start);
    quicksort_u32(cpu_arr, N);
    clock_gettime(CLOCK_MONOTONIC, &t_cpu_end);

    // ----- Allocate & load DPUs -----
    struct dpu_set_t dpu_set, dpu;
    DPU_ASSERT(dpu_alloc(NB_DPUS, "backend=simulator", &dpu_set));
    DPU_ASSERT(dpu_load(dpu_set, DPU_BINARY, NULL));

    // ----- Shard across DPUs -----
    uint32_t *shard_starts = (uint32_t *)malloc(sizeof(uint32_t) * NB_DPUS);
    uint32_t *shard_counts = (uint32_t *)malloc(sizeof(uint32_t) * NB_DPUS);
    if (!shard_starts || !shard_counts) { fprintf(stderr, "OOM\n"); return 1; }

    uint32_t base = 0;
    for (uint32_t i = 0; i < NB_DPUS; i++) {
        uint32_t cnt = N / NB_DPUS + (i < (N % NB_DPUS) ? 1 : 0);
        shard_starts[i] = base;
        shard_counts[i] = cnt;
        base += cnt;
        if (cnt > MAX_ELEMS_PER_DPU) {
            fprintf(stderr, "Shard %u too large for MRAM (%u > %u)\n", i, cnt, MAX_ELEMS_PER_DPU);
            DPU_ASSERT(dpu_free(dpu_set));
            return 1;
        }
    }

    // ----- Copy shards to DPUs & set args -----
    clock_gettime(CLOCK_MONOTONIC, &t_h2d_start);
    uint32_t idx = 0;
    DPU_FOREACH(dpu_set, dpu) {
        const uint32_t start = shard_starts[idx];
        const uint32_t count = shard_counts[idx];

        elem_t *packed = (elem_t *)aligned_alloc(8, sizeof(elem_t) * count);
        if (!packed) { fprintf(stderr, "OOM\n"); return 1; }
        for (uint32_t k = 0; k < count; k++) { packed[k].v = input[start + k]; packed[k].pad = 0; }

        const size_t need_bytes = sizeof(elem_t) * (size_t)count;
        DPU_ASSERT(dpu_copy_to(dpu, "MRAM_ARR", 0, packed, need_bytes));

        struct dpu_args args = { .n_elems = count };
        DPU_ASSERT(dpu_copy_to(dpu, "ARGS", 0, &args, sizeof(args)));

        free(packed);
        idx++;
    }
    clock_gettime(CLOCK_MONOTONIC, &t_h2d_end);

    // ----- Launch -----
    DPU_ASSERT(dpu_launch(dpu_set, DPU_SYNCHRONOUS));
    clock_gettime(CLOCK_MONOTONIC, &t_launch_end);

    // ----- Copy results back -----
    uint32_t **shards = (uint32_t **)malloc(sizeof(uint32_t *) * NB_DPUS);
    if (!shards) { fprintf(stderr, "OOM\n"); return 1; }
    for (uint32_t i = 0; i < NB_DPUS; i++) {
        shards[i] = (uint32_t *)malloc(sizeof(uint32_t) * shard_counts[i]);
        if (!shards[i]) { fprintf(stderr, "OOM\n"); return 1; }
    }

    // pull per-DPU STATS
    struct dpu_stats *stats = (struct dpu_stats *)calloc(NB_DPUS, sizeof(struct dpu_stats));

    idx = 0;
    DPU_FOREACH(dpu_set, dpu) {
        const uint32_t count = shard_counts[idx];

        elem_t *packed = (elem_t *)aligned_alloc(8, sizeof(elem_t) * count);
        if (!packed) { fprintf(stderr, "OOM\n"); return 1; }
        DPU_ASSERT(dpu_copy_from(dpu, "MRAM_ARR", 0, packed, sizeof(elem_t) * count));
        for (uint32_t k = 0; k < count; k++) shards[idx][k] = packed[k].v;
        free(packed);

        if (stats) DPU_ASSERT(dpu_copy_from(dpu, "STATS", 0, &stats[idx], sizeof(struct dpu_stats)));

        idx++;
    }
    clock_gettime(CLOCK_MONOTONIC, &t_d2h_end);

    // ----- Merge shards (k-way) into output -----
    uint32_t *sorted = (uint32_t *)malloc(sizeof(uint32_t) * N);
    heap_node_t *heap = (heap_node_t *)malloc(sizeof(heap_node_t) * NB_DPUS);
    if (!sorted || !heap) { fprintf(stderr, "OOM\n"); return 1; }

    uint32_t heap_n = 0;
    for (uint32_t s = 0; s < NB_DPUS; s++) {
        if (shard_counts[s] > 0) {
            heap[heap_n++] = (heap_node_t){ .value = shards[s][0], .shard_id = s, .idx_in_shard = 0 };
            heap_up(heap, heap_n - 1);
        }
    }
    for (uint32_t out = 0; out < N; out++) {
        heap_node_t top = heap[0];
        sorted[out] = top.value;
        uint32_t s = top.shard_id;
        uint32_t i_in = top.idx_in_shard + 1;
        if (i_in < shard_counts[s]) {
            heap[0] = (heap_node_t){ .value = shards[s][i_in], .shard_id = s, .idx_in_shard = i_in };
            heap_down(heap, heap_n, 0);
        } else {
            heap[0] = heap[--heap_n];
            heap_down(heap, heap_n, 0);
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &t_merge_end);

    // ----- Compare CPU quicksort vs DPU+merge -----
    int ok = (memcmp(cpu_arr, sorted, sizeof(uint32_t) * N) == 0);
    printf("Validation (CPU quicksort vs DPU merge): %s\n", ok ? "PASS" : "FAIL");

    // ----- Print timings -----
    uint64_t ns_cpu   = ns_diff(t_cpu_start,   t_cpu_end);
    uint64_t ns_h2d   = ns_diff(t_h2d_start,   t_h2d_end);
    uint64_t ns_exec  = ns_diff(t_h2d_end,     t_launch_end);
    uint64_t ns_d2h   = ns_diff(t_launch_end,  t_d2h_end);
    uint64_t ns_merge = ns_diff(t_d2h_end,     t_merge_end);
    uint64_t ns_end2end = ns_diff(t_after_rand, t_d2h_end); // after randoms -> data back

    printf("\n--- Timing (ms) ---\n");
    printf("CPU quicksort   : %.3f\n", ns_cpu / 1e6);
    printf("H2D copy        : %.3f\n", ns_h2d / 1e6);
    printf("DPU execute     : %.3f\n", ns_exec / 1e6);
    printf("D2H copy        : %.3f\n", ns_d2h / 1e6);
    printf("Merge           : %.3f\n", ns_merge / 1e6);
    printf("DPU End-to-End  : %.3f  (post-array generated -> end of D2H)\n", ns_end2end / 1e6);

    // summarize DPU cycle stats if available
    if (stats) {
        uint32_t min_c = UINT32_MAX, max_c = 0; double sum_c = 0.0;
        for (uint32_t i = 0; i < NB_DPUS; i++) {
            uint32_t c = stats[i].cycles_sort;
            if (c < min_c) min_c = c;
            if (c > max_c) max_c = c;
            sum_c += (double)c;
        }
        if (NB_DPUS) printf("DPU cycles_sort : min=%u  avg=%.0f  max=%u  (tasklet0)\n",
                            min_c, sum_c / NB_DPUS, max_c);
    }

    // ----- Cleanup -----
    free(sorted);
    for (uint32_t i = 0; i < NB_DPUS; i++) free(shards[i]);
    free(shards);
    free(heap);
    free(stats);
    free(shard_starts);
    free(shard_counts);
    free(cpu_arr);
    free(input);
    DPU_ASSERT(dpu_free(dpu_set));
    return ok ? 0 : 1;
}
