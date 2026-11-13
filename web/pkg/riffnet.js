/* @ts-self-types="./riffnet.d.ts" */

/**
 * Which trained model `RiffNetDetector` runs. Guitar (MIDI 36-90) and bass
 * (MIDI 23-65, a 5-string's low B upward) are separately trained models,
 * not a shared backbone, so switching rebuilds the detector — cheap, since
 * the weights are already embedded in the wasm binary (no I/O, just
 * re-reading static bytes).
 * @enum {0 | 1}
 */
export const Instrument = Object.freeze({
    Guitar: 0, "0": "Guitar",
    Bass: 1, "1": "Bass",
});

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
    __destroy_into_raw() {
        const ptr = this.__wbg_ptr;
        this.__wbg_ptr = 0;
        RiffNetDetectorFinalization.unregister(this);
        return ptr;
    }
    free() {
        const ptr = this.__destroy_into_raw();
        wasm.__wbg_riffnetdetector_free(ptr, 0);
    }
    /**
     * Profiling: time of the last hop's CQT extraction (ms; front-end only,
     * excludes the model forward pass).
     * @returns {number}
     */
    get cqt_ms() {
        const ret = wasm.riffnetdetector_cqt_ms(this.__wbg_ptr);
        return ret;
    }
    /**
     * Which model is active. Setting this rebuilds the detector for the new
     * instrument (thresholds/refractory/max_notes reset to that
     * instrument's defaults; sample rate carries over).
     * @returns {Instrument}
     */
    get instrument() {
        const ret = wasm.riffnetdetector_instrument(this.__wbg_ptr);
        return ret;
    }
    /**
     * Max simultaneous notes reported per frame. Default 6.
     * @returns {number}
     */
    get max_notes() {
        const ret = wasm.riffnetdetector_max_notes(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * * `sample_rate` — microphone sample rate (e.g. 44100.0)
     *
     * Defaults to `Instrument::Guitar`; set `.instrument` to switch.
     * @param {number} sample_rate
     */
    constructor(sample_rate) {
        const ret = wasm.riffnetdetector_new(sample_rate);
        this.__wbg_ptr = ret;
        RiffNetDetectorFinalization.register(this, this.__wbg_ptr, this);
        return this;
    }
    /**
     * Onset probability threshold to fire an onset. Default 0.60.
     * @returns {number}
     */
    get onset_threshold() {
        const ret = wasm.riffnetdetector_onset_threshold(this.__wbg_ptr);
        return ret;
    }
    /**
     * Pitch posterior threshold to report a note. Default 0.50.
     * @returns {number}
     */
    get pitch_threshold() {
        const ret = wasm.riffnetdetector_pitch_threshold(this.__wbg_ptr);
        return ret;
    }
    /**
     * Returns `{onset, notes, chord_name}` for the most recently completed
     * hop, or `null` if no hop has landed since the last poll — callers must
     * keep their previous display state on `null`. This distinction matters:
     * an object with empty `notes` is a real per-hop verdict ("nothing
     * playing", blank the display), while `null` just means "no new data
     * this block" (hops are 256 canonical samples but audio arrives in
     * 128-sample worklet blocks, so roughly every other poll has no hop —
     * treating those as empty verdicts made the display blink).
     * @returns {any}
     */
    poll() {
        const ret = wasm.riffnetdetector_poll(this.__wbg_ptr);
        return takeObject(ret);
    }
    /**
     * Feed a `Float32Array` of audio samples. Call `poll()` afterward.
     * @param {Float32Array} samples
     */
    process_audio(samples) {
        const ptr0 = passArrayF32ToWasm0(samples, wasm.__wbindgen_export2);
        const len0 = WASM_VECTOR_LEN;
        wasm.riffnetdetector_process_audio(this.__wbg_ptr, ptr0, len0);
    }
    /**
     * Minimum frames between onsets (debounce). Default 6.
     * @returns {number}
     */
    get refractory() {
        const ret = wasm.riffnetdetector_refractory(this.__wbg_ptr);
        return ret >>> 0;
    }
    /**
     * @param {Instrument} v
     */
    set instrument(v) {
        wasm.riffnetdetector_set_instrument(this.__wbg_ptr, v);
    }
    /**
     * @param {number} v
     */
    set max_notes(v) {
        wasm.riffnetdetector_set_max_notes(this.__wbg_ptr, v);
    }
    /**
     * @param {number} v
     */
    set onset_threshold(v) {
        wasm.riffnetdetector_set_onset_threshold(this.__wbg_ptr, v);
    }
    /**
     * @param {number} v
     */
    set pitch_threshold(v) {
        wasm.riffnetdetector_set_pitch_threshold(this.__wbg_ptr, v);
    }
    /**
     * @param {number} v
     */
    set refractory(v) {
        wasm.riffnetdetector_set_refractory(this.__wbg_ptr, v);
    }
    /**
     * Play-along verification (known-song mode). Pass the chart's expected
     * MIDI notes for the current beat; returns each note's model confidence
     * in [0,1] (rolling mean over the last ~93 ms of pitch posteriors).
     * Treat confidence >= ~0.30 as a hit; tune ~0.25–0.40 strict↔lenient.
     *
     * JS: `const conf = det.verify_expected(new Int32Array([36, 40, 47, 52]));`
     * @param {Int32Array} expected_midi
     * @returns {Float32Array}
     */
    verify_expected(expected_midi) {
        try {
            const retptr = wasm.__wbindgen_add_to_stack_pointer(-16);
            const ptr0 = passArray32ToWasm0(expected_midi, wasm.__wbindgen_export2);
            const len0 = WASM_VECTOR_LEN;
            wasm.riffnetdetector_verify_expected(retptr, this.__wbg_ptr, ptr0, len0);
            var r0 = getDataViewMemory0().getInt32(retptr + 4 * 0, true);
            var r1 = getDataViewMemory0().getInt32(retptr + 4 * 1, true);
            var v2 = getArrayF32FromWasm0(r0, r1).slice();
            wasm.__wbindgen_export3(r0, r1 * 4, 4);
            return v2;
        } finally {
            wasm.__wbindgen_add_to_stack_pointer(16);
        }
    }
}
if (Symbol.dispose) RiffNetDetector.prototype[Symbol.dispose] = RiffNetDetector.prototype.free;
function __wbg_get_imports() {
    const import0 = {
        __proto__: null,
        __wbg___wbindgen_is_undefined_c05833b95a3cf397: function(arg0) {
            const ret = getObject(arg0) === undefined;
            return ret;
        },
        __wbg___wbindgen_throw_344f42d3211c4765: function(arg0, arg1) {
            throw new Error(getStringFromWasm0(arg0, arg1));
        },
        __wbg_getRandomValues_cc7f052a444bb2ce: function() { return handleError(function (arg0, arg1) {
            globalThis.crypto.getRandomValues(getArrayU8FromWasm0(arg0, arg1));
        }, arguments); },
        __wbg_instanceof_Window_05ba1ee4f6781663: function(arg0) {
            let result;
            try {
                result = getObject(arg0) instanceof Window;
            } catch (_) {
                result = false;
            }
            const ret = result;
            return ret;
        },
        __wbg_new_32b398fb48b6d94a: function() {
            const ret = new Array();
            return addHeapObject(ret);
        },
        __wbg_new_da52cf8fe3429cb2: function() {
            const ret = new Object();
            return addHeapObject(ret);
        },
        __wbg_now_390768da5ee9e776: function(arg0) {
            const ret = getObject(arg0).now();
            return ret;
        },
        __wbg_performance_3ef602e13d6c3b56: function(arg0) {
            const ret = getObject(arg0).performance;
            return isLikeNone(ret) ? 0 : addHeapObject(ret);
        },
        __wbg_set_6be42768c690e380: function(arg0, arg1, arg2) {
            getObject(arg0)[takeObject(arg1)] = takeObject(arg2);
        },
        __wbg_set_8a16b38e4805b298: function(arg0, arg1, arg2) {
            getObject(arg0)[arg1 >>> 0] = takeObject(arg2);
        },
        __wbg_static_accessor_GLOBAL_4ef717fb391d88b7: function() {
            const ret = typeof global === 'undefined' ? null : global;
            return isLikeNone(ret) ? 0 : addHeapObject(ret);
        },
        __wbg_static_accessor_GLOBAL_THIS_8d1badc68b5a74f4: function() {
            const ret = typeof globalThis === 'undefined' ? null : globalThis;
            return isLikeNone(ret) ? 0 : addHeapObject(ret);
        },
        __wbg_static_accessor_SELF_146583524fe1469b: function() {
            const ret = typeof self === 'undefined' ? null : self;
            return isLikeNone(ret) ? 0 : addHeapObject(ret);
        },
        __wbg_static_accessor_WINDOW_f2829a2234d7819e: function() {
            const ret = typeof window === 'undefined' ? null : window;
            return isLikeNone(ret) ? 0 : addHeapObject(ret);
        },
        __wbindgen_cast_0000000000000001: function(arg0) {
            // Cast intrinsic for `F64 -> Externref`.
            const ret = arg0;
            return addHeapObject(ret);
        },
        __wbindgen_cast_0000000000000002: function(arg0, arg1) {
            // Cast intrinsic for `Ref(String) -> Externref`.
            const ret = getStringFromWasm0(arg0, arg1);
            return addHeapObject(ret);
        },
        __wbindgen_object_clone_ref: function(arg0) {
            const ret = getObject(arg0);
            return addHeapObject(ret);
        },
        __wbindgen_object_drop_ref: function(arg0) {
            takeObject(arg0);
        },
    };
    return {
        __proto__: null,
        "./riffnet_bg.js": import0,
    };
}

const RiffNetDetectorFinalization = (typeof FinalizationRegistry === 'undefined')
    ? { register: () => {}, unregister: () => {} }
    : new FinalizationRegistry(ptr => wasm.__wbg_riffnetdetector_free(ptr, 1));

function addHeapObject(obj) {
    if (heap_next === heap.length) heap.push(heap.length + 1);
    const idx = heap_next;
    heap_next = heap[idx];

    heap[idx] = obj;
    return idx;
}

function dropObject(idx) {
    if (idx < 1028) return;
    heap[idx] = heap_next;
    heap_next = idx;
}

function getArrayF32FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getFloat32ArrayMemory0().subarray(ptr / 4, ptr / 4 + len);
}

function getArrayU8FromWasm0(ptr, len) {
    ptr = ptr >>> 0;
    return getUint8ArrayMemory0().subarray(ptr / 1, ptr / 1 + len);
}

let cachedDataViewMemory0 = null;
function getDataViewMemory0() {
    if (cachedDataViewMemory0 === null || cachedDataViewMemory0.buffer.detached === true || (cachedDataViewMemory0.buffer.detached === undefined && cachedDataViewMemory0.buffer !== wasm.memory.buffer)) {
        cachedDataViewMemory0 = new DataView(wasm.memory.buffer);
    }
    return cachedDataViewMemory0;
}

let cachedFloat32ArrayMemory0 = null;
function getFloat32ArrayMemory0() {
    if (cachedFloat32ArrayMemory0 === null || cachedFloat32ArrayMemory0.byteLength === 0) {
        cachedFloat32ArrayMemory0 = new Float32Array(wasm.memory.buffer);
    }
    return cachedFloat32ArrayMemory0;
}

function getStringFromWasm0(ptr, len) {
    return decodeText(ptr >>> 0, len);
}

let cachedUint32ArrayMemory0 = null;
function getUint32ArrayMemory0() {
    if (cachedUint32ArrayMemory0 === null || cachedUint32ArrayMemory0.byteLength === 0) {
        cachedUint32ArrayMemory0 = new Uint32Array(wasm.memory.buffer);
    }
    return cachedUint32ArrayMemory0;
}

let cachedUint8ArrayMemory0 = null;
function getUint8ArrayMemory0() {
    if (cachedUint8ArrayMemory0 === null || cachedUint8ArrayMemory0.byteLength === 0) {
        cachedUint8ArrayMemory0 = new Uint8Array(wasm.memory.buffer);
    }
    return cachedUint8ArrayMemory0;
}

function getObject(idx) { return heap[idx]; }

function handleError(f, args) {
    try {
        return f.apply(this, args);
    } catch (e) {
        wasm.__wbindgen_export(addHeapObject(e));
    }
}

let heap = new Array(1024).fill(undefined);
heap.push(undefined, null, true, false);

let heap_next = heap.length;

function isLikeNone(x) {
    return x === undefined || x === null;
}

function passArray32ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 4, 4) >>> 0;
    getUint32ArrayMemory0().set(arg, ptr / 4);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function passArrayF32ToWasm0(arg, malloc) {
    const ptr = malloc(arg.length * 4, 4) >>> 0;
    getFloat32ArrayMemory0().set(arg, ptr / 4);
    WASM_VECTOR_LEN = arg.length;
    return ptr;
}

function takeObject(idx) {
    const ret = getObject(idx);
    dropObject(idx);
    return ret;
}

let cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
cachedTextDecoder.decode();
const MAX_SAFARI_DECODE_BYTES = 2146435072;
let numBytesDecoded = 0;
function decodeText(ptr, len) {
    numBytesDecoded += len;
    if (numBytesDecoded >= MAX_SAFARI_DECODE_BYTES) {
        cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
        cachedTextDecoder.decode();
        numBytesDecoded = len;
    }
    return cachedTextDecoder.decode(getUint8ArrayMemory0().subarray(ptr, ptr + len));
}

let WASM_VECTOR_LEN = 0;

let wasmModule, wasmInstance, wasm;
function __wbg_finalize_init(instance, module) {
    wasmInstance = instance;
    wasm = instance.exports;
    wasmModule = module;
    cachedDataViewMemory0 = null;
    cachedFloat32ArrayMemory0 = null;
    cachedUint32ArrayMemory0 = null;
    cachedUint8ArrayMemory0 = null;
    return wasm;
}

async function __wbg_load(module, imports) {
    if (typeof Response === 'function' && module instanceof Response) {
        if (typeof WebAssembly.instantiateStreaming === 'function') {
            try {
                return await WebAssembly.instantiateStreaming(module, imports);
            } catch (e) {
                const validResponse = module.ok && expectedResponseType(module.type);

                if (validResponse && module.headers.get('Content-Type') !== 'application/wasm') {
                    console.warn("`WebAssembly.instantiateStreaming` failed because your server does not serve Wasm with `application/wasm` MIME type. Falling back to `WebAssembly.instantiate` which is slower. Original error:\n", e);

                } else { throw e; }
            }
        }

        const bytes = await module.arrayBuffer();
        return await WebAssembly.instantiate(bytes, imports);
    } else {
        const instance = await WebAssembly.instantiate(module, imports);

        if (instance instanceof WebAssembly.Instance) {
            return { instance, module };
        } else {
            return instance;
        }
    }

    function expectedResponseType(type) {
        switch (type) {
            case 'basic': case 'cors': case 'default': return true;
        }
        return false;
    }
}

function initSync(module) {
    if (wasm !== undefined) return wasm;


    if (module !== undefined) {
        if (Object.getPrototypeOf(module) === Object.prototype) {
            ({module} = module)
        } else {
            console.warn('using deprecated parameters for `initSync()`; pass a single object instead')
        }
    }

    const imports = __wbg_get_imports();
    if (!(module instanceof WebAssembly.Module)) {
        module = new WebAssembly.Module(module);
    }
    const instance = new WebAssembly.Instance(module, imports);
    return __wbg_finalize_init(instance, module);
}

async function __wbg_init(module_or_path) {
    if (wasm !== undefined) return wasm;


    if (module_or_path !== undefined) {
        if (Object.getPrototypeOf(module_or_path) === Object.prototype) {
            ({module_or_path} = module_or_path)
        } else {
            console.warn('using deprecated parameters for the initialization function; pass a single object instead')
        }
    }

    if (module_or_path === undefined) {
        module_or_path = new URL('riffnet_bg.wasm', import.meta.url);
    }
    const imports = __wbg_get_imports();

    if (typeof module_or_path === 'string' || (typeof Request === 'function' && module_or_path instanceof Request) || (typeof URL === 'function' && module_or_path instanceof URL)) {
        module_or_path = fetch(module_or_path);
    }

    const { instance, module } = await __wbg_load(await module_or_path, imports);

    return __wbg_finalize_init(instance, module);
}

export { initSync, __wbg_init as default };
