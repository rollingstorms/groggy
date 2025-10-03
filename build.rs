fn main() {
    // Tell Cargo to rerun build if web assets change
    println!("cargo:rerun-if-changed=web/index.html");
    println!("cargo:rerun-if-changed=web/app.js");
    println!("cargo:rerun-if-changed=web/styles.css");
}
