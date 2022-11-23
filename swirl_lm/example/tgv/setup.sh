PYTHON_PACKAGES=$(pip show swirl_lm | sed -n -e 's/Location: \(.*\)/\1/p')
proto_names=(
    'base/parameters.proto'
    'boundary_condition/boundary_conditions.proto'
    'boundary_condition/boundary_models.proto'
    'boundary_condition/immersed_boundary_method.proto'
    'boundary_condition/monin_obukhov_similarity_theory.proto'
    'boundary_condition/rayleigh_damping_layer.proto'
    'equations/pressure.proto'
    'equations/scalars.proto'
    'linalg/poisson_solver.proto'
    'numerics/numerics.proto'
    'physics/combustion/combustion.proto'
    'physics/combustion/wood.proto'
    'physics/thermodynamics/thermodynamics.proto'
    'utility/grid_parametrization.proto'
    'utility/monitor.proto'
    'utility/probe.proto'
)

for proto in ${proto_names[@]}; do
  protoc -I=$PYTHON_PACKAGES --python_out=$PYTHON_PACKAGES \
    $PYTHON_PACKAGES/swirl_lm/$proto
done
