fn main() {
    println!("cargo:rustc-check-cfg=cfg(Py_LIMITED_API)");
    pyo3_build_config::use_pyo3_cfgs();
    pyo3_build_config::add_extension_module_link_args();
}
