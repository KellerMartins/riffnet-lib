use crate::DetectedNote;

const NOTE_NAMES: [&str; 12] = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];

struct ChordTemplate {
    name: &'static str,
    intervals: &'static [i32],
}

// Define the templates
const TEMPLATES: [ChordTemplate; 8] = [
    ChordTemplate { name: "maj", intervals: &[0, 4, 7] },       // Major
    ChordTemplate { name: "min", intervals: &[0, 3, 7] },       // Minor
    ChordTemplate { name: "5", intervals: &[0, 7] },            // Power Chord
    ChordTemplate { name: "maj7", intervals: &[0, 4, 7, 11] },   // Major 7th
    ChordTemplate { name: "min7", intervals: &[0, 3, 7, 10] },   // Minor 7th
    ChordTemplate { name: "7", intervals: &[0, 4, 7, 10] },      // Dominant 7th
    ChordTemplate { name: "sus4", intervals: &[0, 5, 7] },       // Suspended 4th
    ChordTemplate { name: "sus2", intervals: &[0, 2, 7] },       // Suspended 2nd
];

pub fn recognize_chord(peaks: &[DetectedNote]) -> Option<String> {
    if peaks.is_empty() {
        return None;
    }

    // Extract unique pitch classes from the top peaks (up to 6 peaks)
    let mut detected_pcs = Vec::new();
    for peak in peaks.iter().take(6) {
        let midi_round = peak.midi_note.round() as i32;
        let pc = ((midi_round % 12) + 12) % 12;
        if !detected_pcs.contains(&pc) {
            detected_pcs.push(pc);
        }
    }

    if detected_pcs.len() < 2 {
        return None;
    }

    let mut best_score = -99.0;
    let mut best_chord = None;

    // Check all possible roots (0 to 11)
    for root in 0..12 {
        for template in &TEMPLATES {
            let mut matched_count = 0;
            let mut template_pcs = Vec::new();

            for &interval in template.intervals {
                let pc = (root + interval) % 12;
                template_pcs.push(pc);
                if detected_pcs.contains(&pc) {
                    matched_count += 1;
                }
            }

            // Count extra notes (notes we detected that are not in the template)
            let mut extra_count = 0;
            for &pc in &detected_pcs {
                if !template_pcs.contains(&pc) {
                    extra_count += 1;
                }
            }

            // Score function:
            // 1. Must match at least the root or a large portion of the template
            // 2. Heavy penalty if the root is not present
            let root_present = detected_pcs.contains(&root);
            let root_penalty = if root_present { 0.0 } else { -2.0 };

            // Jaccard-like scoring with preference for complete templates
            let completion_bonus = if matched_count == template.intervals.len() { 2.0 } else { 0.0 };
            
            let score = matched_count as f32 
                - (extra_count as f32 * 0.4) 
                + completion_bonus 
                + root_penalty;

            if score > best_score && matched_count >= 2 {
                best_score = score;
                let chord_suffix = if template.name == "maj" { "" } else { template.name };
                best_chord = Some(format!("{}{}", NOTE_NAMES[root as usize], chord_suffix));
            }
        }
    }

    // Only return if the score is reasonably high
    if best_score >= 1.2 {
        best_chord
    } else {
        None
    }
}
