// mergesort_dpu.c
#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#include <defs.h>
#include <mram.h>
#include <alloc.h>
#include <barrier.h>
#include <perfcounter.h>

#ifndef NR_TASKLETS
#define NR_TASKLETS 16
#endif

// ---------- Stats & sync ----------
struct dpu_stats {
    uint32_t n_elems;
    uint32_t nr_tasklets;
    uint32_t cycles_sort;    // cycles during mergesort region (tasklet 0)
    uint32_t cycles_total;   // cycles since perfcounter_config (tasklet 0)
};
__host struct dpu_stats STATS;

BARRIER_INIT(sync_barrier, NR_TASKLETS);

// ---------- Element & MRAM layout ----------
typedef struct __attribute__((packed, aligned(8))) {
    uint32_t v;
    uint32_t pad;
} elem_t;

#ifndef MAX_ELEMS_PER_DPU
#define MAX_ELEMS_PER_DPU (1u << 20)
#endif

__mram_noinit elem_t MRAM_ARR[MAX_ELEMS_PER_DPU];
__mram_noinit elem_t MRAM_TMP[MAX_ELEMS_PER_DPU];  // workspace for ping-pong

// ---------- Host-provided arguments ----------
struct dpu_args { uint32_t n_elems; };
__host struct dpu_args ARGS;

// ---------- MRAM helpers ----------
static inline void mram_read_elem_src(const elem_t *base, uint32_t idx, elem_t *dst) {
    mram_read((void *)&base[idx], dst, sizeof(elem_t));
}
static inline void mram_write_elem_dst(elem_t *base, uint32_t idx, const elem_t *src) {
    mram_write((void *)src, (void *)&base[idx], sizeof(elem_t));
}

// Merge [lo, mid) and [mid, hi) from SRC -> DST (all indices 0-based)
static void merge_run(const elem_t *SRC, elem_t *DST, uint32_t lo, uint32_t mid, uint32_t hi) {
    uint32_t i = lo, j = mid, k = lo;
    elem_t a, b;

    // pre-read the first elements (guard for empty runs)
    bool has_a = (i < mid);
    bool has_b = (j < hi);
    if (has_a) mram_read_elem_src(SRC, i, &a);
    if (has_b) mram_read_elem_src(SRC, j, &b);

    while (has_a && has_b) {
        if (a.v <= b.v) {
            mram_write_elem_dst(DST, k++, &a);
            i++;
            has_a = (i < mid);
            if (has_a) mram_read_elem_src(SRC, i, &a);
        } else {
            mram_write_elem_dst(DST, k++, &b);
            j++;
            has_b = (j < hi);
            if (has_b) mram_read_elem_src(SRC, j, &b);
        }
    }
    while (has_a) {
        mram_write_elem_dst(DST, k++, &a);
        i++;
        has_a = (i < mid);
        if (has_a) mram_read_elem_src(SRC, i, &a);
    }
    while (has_b) {
        mram_write_elem_dst(DST, k++, &b);
        j++;
        has_b = (j < hi);
        if (has_b) mram_read_elem_src(SRC, j, &b);
    }
}

// Bottom-up mergesort from SRC -> DST using ping-pong passes.
// After the final pass, if result sits in TMP, copy back to ARR.
static void mergesort_mram(uint32_t n) {
    if (n <= 1) return;

    const elem_t *src = MRAM_ARR;   // start reading from ARR
    elem_t *dst = MRAM_TMP;         // writing to TMP

    // width = size of each sorted run in the current pass
    for (uint32_t width = 1; width < n; width <<= 1) {
        uint32_t lo = 0;
        while (lo < n) {
            uint32_t mid = lo + width;
            uint32_t hi  = lo + (width << 1);
            if (mid > n) mid = n;
            if (hi  > n) hi  = n;

            // If the right run is empty, just copy the left run
            if (mid >= hi) {
                // linear copy lo..mid-1 from src to dst
                for (uint32_t t = lo; t < mid; t++) {
                    elem_t tmp;
                    mram_read_elem_src(src, t, &tmp);
                    mram_write_elem_dst(dst, t, &tmp);
                }
            } else {
                merge_run(src, dst, lo, mid, hi);
            }
            lo += (width << 1);
        }
        // swap roles
        const elem_t *old_src = src;
        src = (const elem_t *)dst;
        dst = (elem_t *)old_src;
    }

    // If final data landed in MRAM_TMP, copy back to MRAM_ARR
    if (src == (const elem_t *)MRAM_TMP) {
        for (uint32_t t = 0; t < n; t++) {
            elem_t tmp;
            mram_read_elem_src(MRAM_TMP, t, &tmp);
            mram_write_elem_dst(MRAM_ARR, t, &tmp);
        }
    }
}

// ---------- Tasklet entry ----------
int main() {
    if (me() == 0) {
        mem_reset();
        perfcounter_config(COUNT_CYCLES, true);
        STATS.n_elems = ARGS.n_elems;
        STATS.nr_tasklets = NR_TASKLETS;
        STATS.cycles_sort = 0;
        STATS.cycles_total = 0;
    }
    barrier_wait(&sync_barrier);

    uint32_t n = ARGS.n_elems;

    uint32_t start_c = 0, end_c = 0;
    if (me() == 0) start_c = perfcounter_get();
    barrier_wait(&sync_barrier);

    if (me() == 0 && n > 1) {
        // tasklet 0 mergesorts the whole shard
        mergesort_mram(n);
    }
    barrier_wait(&sync_barrier);

    if (me() == 0) {
        end_c = perfcounter_get();
        STATS.cycles_sort  = end_c - start_c;
        STATS.cycles_total = perfcounter_get();
    }
    barrier_wait(&sync_barrier);
    return 0;
}
