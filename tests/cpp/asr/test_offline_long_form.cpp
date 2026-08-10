// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include <cstdio>
#include <vector>

#include "runner.h"

using namespace nemo_speech::asr;

static int g_fail = 0;

static void
check(bool ok, const char* what) {
    std::fprintf(stdout, "[%s] %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok)
        g_fail++;
}

static void
test_activation_and_geometry() {
    using offline_detail::make_windows;

    const auto disabled = make_windows(300000, 0, 3500, 180000);
    check(
        disabled.size() == 1 && disabled[0].input_samples == 300000 &&
            disabled[0].owned_samples == 300000,
        "offline long-form: zero chunk duration disables bounded windows");

    const auto short_audio = make_windows(179999, 20000, 3500, 180000);
    check(
        short_audio.size() == 1 && short_audio[0].input_samples == 179999 &&
            short_audio[0].owned_samples == 179999,
        "offline long-form: audio below activation stays full utterance");

    const auto windows = make_windows(180000, 20000, 3500, 180000);
    check(windows.size() == 9, "offline long-form: threshold activates fixed windows");
    check(
        windows.front().input_offset == 0 && windows.front().input_samples == 27000 &&
            windows.front().owned_offset == 0 && windows.front().owned_samples == 20000,
        "offline long-form: first window owns the leading chunk");
    check(
        windows[1].input_offset == 16500 && windows[1].owned_offset == 3500 &&
            windows[1].owned_samples == 20000,
        "offline long-form: middle window has symmetric context");
    check(
        windows.back().input_offset == 153000 && windows.back().owned_offset == 7000 &&
            windows.back().owned_samples == 20000,
        "offline long-form: final fixed window shifts left without changing ownership");

    const auto partial_tail = make_windows(185000, 20000, 3500, 180000);
    check(
        partial_tail.size() == 10 && partial_tail.back().input_offset == 158000 &&
            partial_tail.back().input_samples == 27000 &&
            partial_tail.back().owned_offset == 22000 && partial_tail.back().owned_samples == 5000,
        "offline long-form: partial tail keeps fixed input shape and exact ownership");
}

static void
test_shift_geometry() {
    const auto window = offline_detail::make_windows(180000, 20000, 3500, 180000)[1];
    const auto left = offline_detail::shift_window(window, 180000, -1000);
    const auto right = offline_detail::shift_window(window, 180000, 1000);
    check(
        left.input_offset == 15500 && left.owned_offset == 4500,
        "offline retry: left shift preserves the owned interval");
    check(
        right.input_offset == 17500 && right.owned_offset == 2500,
        "offline retry: right shift preserves the owned interval");

    const auto first = offline_detail::make_windows(180000, 20000, 3500, 180000).front();
    check(
        offline_detail::shift_window(first, 180000, -1000).input_offset == 0 &&
            offline_detail::shift_window(first, 180000, 1000).input_offset == 0,
        "offline retry: shifts clamp at the recording boundary");
}

static void
test_emission_gap_trigger() {
    std::vector<WordTiming> continuous = {
        {.word = "a", .start_frame = 0, .end_frame = 5},
        {.word = "b", .start_frame = 8, .end_frame = 13},
        {.word = "c", .start_frame = 16, .end_frame = 20},
    };
    check(
        !offline_detail::has_large_emission_gap(continuous, 0, 20, 7),
        "offline retry: ordinary word spacing does not trigger");

    std::vector<WordTiming> collapsed = {
        {.word = "a", .start_frame = 0, .end_frame = 4},
        {.word = "b", .start_frame = 5, .end_frame = 8},
    };
    check(
        offline_detail::has_large_emission_gap(collapsed, 0, 20, 7),
        "offline retry: long trailing emission gap triggers");
    check(
        offline_detail::has_large_emission_gap({}, 0, 20, 7),
        "offline retry: empty owned output triggers");
}

static void
test_consensus_acceptance() {
    check(
        offline_detail::materially_more_words(20, 30) &&
            !offline_detail::materially_more_words(20, 29),
        "offline retry: recovery requires eight words and fifty percent");
    check(
        offline_detail::select_consensus_retry(20, {40, 41}) == 1,
        "offline retry: two recoveries select the consensus candidate");
    check(
        offline_detail::select_consensus_retry(20, {40, 20}) == -1,
        "offline retry: one high-count outlier is rejected");
    check(
        offline_detail::select_consensus_retry(20, {40, 80}) == -1,
        "offline retry: divergent high-count candidates are rejected");
    check(
        offline_detail::select_consensus_retry(20, {29, 30, 31}) == 2,
        "offline retry: only materially larger candidates form consensus");
}

int
main() {
    test_activation_and_geometry();
    test_shift_geometry();
    test_emission_gap_trigger();
    test_consensus_acceptance();
    std::fprintf(
        stdout, "\n%s (%d failure%s)\n", g_fail ? "FAILED" : "ALL PASSED", g_fail,
        g_fail == 1 ? "" : "s");
    return g_fail ? 1 : 0;
}
