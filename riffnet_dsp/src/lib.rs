pub mod dsp;
pub mod chord;

pub use dsp::CqtFrontEnd;

/// A single detected note (one peak of the model's pitch posterior).
#[derive(Debug, Clone)]
pub struct DetectedNote {
    pub midi_note: f32,
    pub frequency: f32,
    pub amplitude: f32,
}

/// Result of decoding one model frame (see `riffnet`'s `decode_logits`).
#[derive(Debug, Clone)]
pub struct DetectionResult {
    pub onset_detected: bool,
    pub peaks: Vec<DetectedNote>,
    pub chord_name: Option<String>,
}
