#! /bin/bash

# PROTOC_URL points to a protoc binary that is compatible with the protobuf
# runtime. Swirl-LM uses the protobuf runtime that is installed by
# TensorFlow. This runtime is not compatible with the latest protoc as of
# January 2023, which is why we explicitly download a specific version. For
# TensorFlow's current protobuf runtime version, see:
#
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/ci_build/release/requirements_common.txt
#
# TensorFlow also installs protoc, which we could also try to use instead of
# installing our own version, but that binary is not necessarily accessible to
# the account setting up Swirl-LM.
PROTOC_URL=https://github.com/protocolbuffers/protobuf/releases/download/v3.19.6/protoc-3.19.6-linux-x86_64.zip

# Path to local copy of protoc, relative to Swirl-LM.
PROTOC=protoc/bin/protoc

function install_protoc() (
  # Downloads protoc binary if it's not already downloaded.
  if [ ! -x "$PROTOC" ]; then
    echo "Installing protoc."
    mkdir protoc
    cd protoc
    curl -L -o protoc.zip "$PROTOC_URL"
    unzip protoc.zip
  else
    echo "Using existing protoc."
  fi
)

function run_protoc() (
  # Generates _pb2 files in the source directories.
  echo "Compiling proto files."
  PROTO_NAMES=$(find swirl_lm -name '*.proto')
  for proto in ${PROTO_NAMES}; do
    "$PROTOC" -I=. --python_out=. $proto
  done
)

function install_swirl_lm() {
  echo "Installing swirl-lm."
  python3 -m pip uninstall -y swirl-lm
  python3 -m pip install .
}

cd $(dirname "$0")/..  # cd to parent of swirl_lm.
install_protoc
run_protoc
install_swirl_lm
