PYTHON_PACKAGES=$(pip show swirl_lm | sed -n -e 's/Location: \(.*\)/\1/p')
proto_names=$(find $PYTHON_PACKAGES/swirl_lm/$proto -name '*.proto')
for proto in ${proto_names}; do
  protoc -I=$PYTHON_PACKAGES --python_out=$PYTHON_PACKAGES $proto
done
