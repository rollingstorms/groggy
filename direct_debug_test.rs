// Direct test with debug instrumentation
use std::process::Command;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Running direct debug test...");

    let test_script = r#"
use groggy::storage::advanced_matrix::neural::autodiff::AutoDiffTensor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Direct Debug Test ===");

    let x = AutoDiffTensor::from_data(vec![1.5], (1, 1), true)?;
    let two = AutoDiffTensor::from_data(vec![2.0], (1, 1), false)?;

    println!("About to call x.multiply(&two)...");
    let u = x.multiply(&two)?;
    println!("Multiply completed, now calling backward...");

    u.backward()?;

    if let Some(x_grad) = x.grad() {
        let x_grad_val: f64 = x_grad.get(0, 0)?.into();
        println!("Final gradient: {}", x_grad_val);
    }

    Ok(())
}
"#;

    std::fs::write("direct_debug_internal.rs", test_script)?;

    println!("Compiling direct debug test...");
    let compile_output = Command::new("rustc")
        .args(&["--edition", "2021", "-L", "target/debug/deps",
               "direct_debug_internal.rs", "--extern", "groggy=target/debug/libgroggy.rlib",
               "-o", "direct_debug_internal"])
        .output()?;

    if !compile_output.status.success() {
        println!("Compilation failed:");
        println!("{}", String::from_utf8_lossy(&compile_output.stderr));
        return Err("Compilation failed".into());
    }

    println!("Running direct debug test...\n");
    let run_output = Command::new("./direct_debug_internal")
        .output()?;

    println!("STDOUT:");
    println!("{}", String::from_utf8_lossy(&run_output.stdout));
    println!("STDERR:");
    println!("{}", String::from_utf8_lossy(&run_output.stderr));

    // Cleanup
    std::fs::remove_file("direct_debug_internal.rs").ok();
    std::fs::remove_file("direct_debug_internal").ok();

    Ok(())
}