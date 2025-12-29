# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CoreGrad is a bare-metal autograd engine in Rust, designed as an educational project to understand how modern ML frameworks work. It prioritizes architectural purity over API compatibility.

**Core Philosophy**:
- **Lazy execution**: Operations build a DAG; computation happens only on `.realize()` or `.backward()`
- **Minimal primitives**: ~8 core ops (Add, Mul, MatMul, Neg, Exp, Log, Sum, Input). Everything else is composition.
- **Arena allocator**: Graph is a `Vec<Node>` with stable indices. Tensors are lightweight handles (`usize` indices).
- **Separate storage**: Node data/gradients stored separately from graph structure to avoid waste on unrealized nodes.

## Project Management

**All work should be tracked in Linear:**
- **Project**: CoreGrad (https://linear.app/wx02shi/project/coregrad-f71313a513bc)
- **Team**: Wx02shi
- Issues WX0-52 through WX0-75 contain the full implementation roadmap

**For Claude Code sessions:**
- **Use Linear MCP** to create, update, and track issues throughout development
- Reference Linear issue IDs (e.g., WX0-52) when working on tasks
- Update issue status as work progresses (Backlog → In Progress → Done)
- Create new issues for bugs, enhancements, or unforeseen work
- Use Linear to maintain project visibility and continuity across sessions

## Build Commands

```bash
# Build the library
cargo build

# Run tests
cargo test

# Run tests with output
cargo test -- --nocapied

# Build in release mode (for benchmarking)
cargo build --release

# Check without building
cargo check
```

## Architecture

### Graph Structure (Arena Pattern)

The computation graph uses an **arena allocator**:
- `Graph` owns a `Vec<Node>`
- `Tensor` is just a wrapper around `usize` (index into the graph)
- Nodes are append-only and never move (stable IDs)
- Data and gradients stored in separate `HashMap<NodeId, Array>` to save memory

### Execution Model

1. **Graph Building Phase**: User creates tensors and operations → builds DAG
2. **Realization Phase**: User calls `.realize()` → executes forward pass, populates data storage
3. **Backward Phase**: User calls `.backward()` → topological sort + reverse-mode autodiff

**Important**: `.backward()` does NOT auto-realize. User must call `.realize()` first.

### Primitive Operations

The engine implements exactly 8 primitive ops:

**Unary**: `Neg`, `Exp`, `Log`, `Sum` (reduce)
**Binary**: `Add`, `Mul`, `MatMul`
**Special**: `Input` (leaf node)

Higher-level ops (ReLU, Sigmoid, Softmax, CrossEntropy) are built by composing primitives.

### Backend Strategy (Future)

CPU backend uses `ndarray` for storage. Matmul uses `ndarray`'s `.dot()` (acceptable for learning; can be optimized with `ndarray-linalg` later).

Metal/GPU backend deferred to Phase 2.

## Design Constraints

- **No broadcasting** (initially): Shapes must match exactly. Reduces complexity in backward pass.
- **f32 only**: No f64, int, or quantization. Type complexity deferred.
- **No eager execution**: This is a JIT compiler, not PyTorch's eager mode.
- **Explicit over convenient**: Verbose API is acceptable. Focus is on engine correctness, not UX.

## Development Notes

- The `Cargo.toml` specifies `path = "src/lib.rs"` - this is a library crate, not a binary
- `edition = "2024"` should be `edition = "2021"` (fix this if it causes issues)
- Tests use `approx` crate for floating-point gradient comparisons
- Benchmark against PyTorch eager mode as the success metric (we should be 5-10x faster due to laziness)
