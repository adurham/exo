#!/bin/bash
set -e

echo "Generating Python code from proto files..."

# Create output directory if it doesn't exist
mkdir -p src/exo/generated

# Generate Python code from protos using uv's python
uv run python -m grpc_tools.protoc \
  --python_out=src/exo/generated \
  --grpc_python_out=src/exo/generated \
  --pyi_out=src/exo/generated \
  -Iproto \
  proto/*.proto

# Make the generated directory a Python package
touch src/exo/generated/__init__.py

echo "✓ Proto generation complete"
echo "Generated files in src/exo/generated/"
ls -la src/exo/generated/
