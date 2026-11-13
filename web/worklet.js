// AudioWorkletProcessor — runs on the audio thread.
// Buffers incoming microphone samples and transfers them to the main thread
// once per render quantum (128 samples) for minimum latency.
class MicProcessor extends AudioWorkletProcessor {
    process(inputs) {
        const ch = inputs[0]?.[0];
        if (ch && ch.length > 0) {
            // Transfer a copy to the main thread
            const copy = new Float32Array(ch);
            this.port.postMessage(copy, [copy.buffer]);
        }
        return true; // keep processor alive
    }
}

registerProcessor('mic-processor', MicProcessor);
