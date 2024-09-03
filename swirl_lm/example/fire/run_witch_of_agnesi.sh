TPU="node-1"
GCS_DIR="gs://swirllm_bucket/witch_of_agnesi/non_reacting_flow"
SURFACE_TYPE=WITCHOFAGNESI2D # or: FILE | RAMP | BUMP
TERRAIN_FILE=None # swirl-lm/swirl_lm/example/fire/tubbs_1k_20kmx20km.ser

python3 ./fire_main.py \
  --simulation_debug=true \
  --c_d=0.01 --dt=0.01 \
  --flat_surface_ignite=False --flat_surface_include_fire=False \
  --flat_surface_init_bl=True \
  --flat_surface_blasius_bl_distance=100000000.0 \
  --flat_surface_blasius_bl_transition=False --flat_surface_blasius_bl_fraction=0.5 \
  --flat_surface_terrain_type=${SURFACE_TYPE} --flat_surface_turbulent_inflow=False \
  --flat_surface_include_coriolis_force=True \
  --flat_surface_use_dynamic_igniter=False \
  --fuel_bed_height=0.0 --fuel_density=0.0 \
  --kernel_size=16 \
  --loading_step=0 \
  --lx=6000.0 --ly=6000.0 --lz=4500.0 \
  --moisture_density=0.0 \
  --num_boundary_points=0 \
  --nx=64 --ny=64 --nz=64 \
  --cx=2 --cy=2 --cz=2 \
  --opt_filter_fuel_density=False \
  --start_step=0 \
  --terrain_filepath=${TERRAIN_FILE} \
  --use_geopotential=False \
  --t_init=283.0 \
  --y_o_init=0.21 \
  --velocity_init_inflow_opt=FROM_COMPONENT_FLAGS \
  --u_init=10.0 --v_init=0.0 \
  --data_dump_prefix=${GCS_DIR} \
  --config_filepath=./witch_of_agnesi.pbtxt \
  --target=$TPU
