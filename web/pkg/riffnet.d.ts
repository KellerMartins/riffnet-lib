/* tslint:disable */
/* eslint-disable */

/**
 * Which trained model `RiffNetDetector` runs. Guitar (MIDI 36-90) and bass
 * (MIDI 23-65, a 5-string's low B upward) are separately trained models,
 * not a shared backbone, so switching rebuilds the detector — cheap, since
 * the weights are already embedded in the wasm binary (no I/O, just
 * re-reading static bytes).
 */
export enum Instrument {
    Guitar = 0,
    Bass = 1,
}

/**
 * Real-time note detector — the full learned model (a ~200-250K-param CNN,
 * depending on instrument), running entirely in Rust/WASM via a hand-rolled
 * streaming forward pass
 * (see `streaming.rs`). No `onnxruntime` dependency, no JS-driven inference
 * call: `process_audio` does everything internally, `poll()` reads the
 * result. Continuously re-evaluates every hop (~5.8ms), so pitch estimates
 * are always current, converging to a confident read ~35ms after a note
 * starts (rolling mean over recent frames).
 */
export class RiffNetDetector {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * * `sample_rate` — microphone sample rate (e.g. 44100.0)
     *
     * Defaults to `Instrument::Guitar`; set `.instrument` to switch.
     */
    constructor(sample_rate: number);
    /**
     * Returns `{onset, notes, chord_name}` for the most recently completed
     * hop, or `null` if no hop has landed since the last poll — callers must
     * keep their previous display state on `null`. This distinction matters:
     * an object with empty `notes` is a real per-hop verdict ("nothing
     * playing", blank the display), while `null` just means "no new data
     * this block" (hops are 256 canonical samples but audio arrives in
     * 128-sample worklet blocks, so roughly every other poll has no hop —
     * treating those as empty verdicts made the display blink).
     */
    poll(): any;
    /**
     * Feed a `Float32Array` of audio samples. Call `poll()` afterward.
     */
    process_audio(samples: Float32Array): void;
    /**
     * Play-along verification (known-song mode). Pass the chart's expected
     * MIDI notes for the current beat; returns each note's model confidence
     * in [0,1] (rolling mean over the last ~93 ms of pitch posteriors).
     * Treat confidence >= ~0.30 as a hit; tune ~0.25–0.40 strict↔lenient.
     *
     * JS: `const conf = det.verify_expected(new Int32Array([36, 40, 47, 52]));`
     */
    verify_expected(expected_midi: Int32Array): Float32Array;
    /**
     * Profiling: time of the last hop's CQT extraction (ms; front-end only,
     * excludes the model forward pass).
     */
    readonly cqt_ms: number;
    /**
     * Which model is active. Setting this rebuilds the detector for the new
     * instrument (thresholds/refractory/max_notes reset to that
     * instrument's defaults; sample rate carries over).
     */
    instrument: Instrument;
    /**
     * Max simultaneous notes reported per frame. Default 6.
     */
    max_notes: number;
    /**
     * Onset probability threshold to fire an onset. Default 0.60.
     */
    onset_threshold: number;
    /**
     * Pitch posterior threshold to report a note. Default 0.50.
     */
    pitch_threshold: number;
    /**
     * Minimum frames between onsets (debounce). Default 6.
     */
    refractory: number;
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_riffnetdetector_free: (a: number, b: number) => void;
    readonly riffnetdetector_cqt_ms: (a: number) => number;
    readonly riffnetdetector_instrument: (a: number) => number;
    readonly riffnetdetector_max_notes: (a: number) => number;
    readonly riffnetdetector_new: (a: number) => number;
    readonly riffnetdetector_onset_threshold: (a: number) => number;
    readonly riffnetdetector_pitch_threshold: (a: number) => number;
    readonly riffnetdetector_poll: (a: number) => number;
    readonly riffnetdetector_process_audio: (a: number, b: number, c: number) => void;
    readonly riffnetdetector_refractory: (a: number) => number;
    readonly riffnetdetector_set_instrument: (a: number, b: number) => void;
    readonly riffnetdetector_set_max_notes: (a: number, b: number) => void;
    readonly riffnetdetector_set_onset_threshold: (a: number, b: number) => void;
    readonly riffnetdetector_set_pitch_threshold: (a: number, b: number) => void;
    readonly riffnetdetector_set_refractory: (a: number, b: number) => void;
    readonly riffnetdetector_verify_expected: (a: number, b: number, c: number, d: number) => void;
    readonly rust_zstd_wasm_shim_calloc: (a: number, b: number) => number;
    readonly rust_zstd_wasm_shim_free: (a: number) => void;
    readonly rust_zstd_wasm_shim_malloc: (a: number) => number;
    readonly rust_zstd_wasm_shim_memcmp: (a: number, b: number, c: number) => number;
    readonly rust_zstd_wasm_shim_memcpy: (a: number, b: number, c: number) => number;
    readonly rust_zstd_wasm_shim_memmove: (a: number, b: number, c: number) => number;
    readonly rust_zstd_wasm_shim_memset: (a: number, b: number, c: number) => number;
    readonly rust_zstd_wasm_shim_qsort: (a: number, b: number, c: number, d: number) => void;
    readonly __wbindgen_export: (a: number) => void;
    readonly __wbindgen_export2: (a: number, b: number) => number;
    readonly __wbindgen_add_to_stack_pointer: (a: number) => number;
    readonly __wbindgen_export3: (a: number, b: number, c: number) => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
