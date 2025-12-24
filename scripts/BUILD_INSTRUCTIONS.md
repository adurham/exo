# Build and Distribution Instructions

## Prerequisites

This project requires Rust nightly toolchain to build the Python bindings.

### Install Rust Nightly

If you don't have rustup installed:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup default nightly
```

If you already have rustup:

```bash
rustup toolchain install nightly
rustup default nightly
```

## Building and Distributing

1. **Build Rust components locally and distribute to all nodes:**
   ```bash
   ./scripts/build-and-distribute.sh
   ```

2. **Deploy the cluster:**
   ```bash
   ./scripts/deploy-static-cluster.sh
   ```

   Or skip the build check if you've already built:
   ```bash
   SKIP_BUILD=yes ./scripts/deploy-static-cluster.sh
   ```

## What the build script does

1. Builds the Rust extension module (`exo_pyo3_bindings`) using `cargo +nightly build --release`
2. Copies the built `.dylib` (macOS) or `.so` (Linux) file to each node
3. Installs it in the Python site-packages directory on each node

This avoids needing to build Rust components on each node, which requires:
- Rust nightly toolchain
- All Rust dependencies
- Time to compile

