use burn_onnx::ModelGen;

fn main() {
    println!("cargo:rerun-if-changed=../model/weights/riffnet.onnx");
    ModelGen::new()
        .input("../model/weights/riffnet.onnx")
        .out_dir("model/")
        .run_from_script();

    println!("cargo:rerun-if-changed=../model/weights/bass_riffnet.onnx");
    ModelGen::new()
        .input("../model/weights/bass_riffnet.onnx")
        .out_dir("model/bass/")
        .run_from_script();
}
