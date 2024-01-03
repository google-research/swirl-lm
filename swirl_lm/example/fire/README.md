# Tubbs Fire on Cloud TPU

## Steps to run the fire simulation on TPU
This example simulates the Tubbs Fire.

1. Create TPUs using the TPU VM architecture and pod software. For example,


   ```sh
   gcloud compute tpus tpu-vm create $TPU --project=$PROJECT \
      --zone=$ZONE \
      --accelerator-type=$TYPE --version=tpu-vm-tf-2.11.0-pod \
      --metadata="tensorflow-env-vars=-e LIBTPU_INIT_ARGS=$TPU_FLAGS"
   ```

1. SSH into the VM:

   ```sh
   gcloud compute ssh --zone="$ZONE" "$TPU"
   ```

   Run the next set of steps on the VM.

1. (VM) Clone Swirl-LM from github:

   ```sh
   git clone https://github.com/google-research/swirl-lm.git
   ```

1. (VM) Install Swirl-LM.

   ```sh
   ./swirl-lm/swirl_lm/setup.sh
   ```

   Note: this uses the files in the local repo and installs the package in the
   user directory and not in site-packages.

1. (VM) Start up `screen` so that if the ssh session is lost, the simulation
   continues in the background and you can connect to the same `screen` terminal
   again. There are other ways to achieve the same goal of being able to
   reconnect to a shell after losing network connectivity (e.g., tmux) so feel
   free to use some other tool if you prefer.

   ```sh
   screen
   ```

1. (VM) Run the solver.

   ```sh
   python3 swirl-lm/swirl_lm/example/fire/fire_main.py \
     --apply_data_load_filter=False --apply_postprocess=True \
     --apply_preprocess=False \
     --c_d=0.01 --dt=0.025 \
     --flat_surface_ignite=True --flat_surface_include_fire=True \
     --flat_surface_terrain_type=FILE --flat_surface_turbulent_inflow=True \
     --flat_surface_use_dynamic_igniter=False \
     --fuel_bed_height=9.785 --fuel_density=0.104 \
     --ignition_center_x=1750.0 --ignition_center_y=12250.0 \
     --ignition_line_angle=90.0 --ignition_line_length=500.0 \
     --ignition_line_thickness=500.0 --ignition_option=SLANT_LINE \
     --ignition_temperature=600.0 \
     --inflow_x_lx=100.0 --inflow_x_ly=100.0 --inflow_x_lz=20.0 \
     --kernel_size=16 \
     --loading_step=0 \
     --lx=20000.0 --ly=20000.0 --lz=2000.0 \
     --moisture_density=0.01 \
     --num_boundary_points=0 \
     --num_cycles=60 --num_steps=4000 \
     --nx=512 --ny=512 --nz=20 \
     --cx=2 --cy=2 --cz=32 \
     --opt_filter_fuel_density=False \
     --postprocess_step_id=100000 \
     --start_step=0 \
     --t_init=283.0 \
     --terrain_filepath=swirl-lm/swirl_lm/example/fire/tubbs_1k_20kmx20km.ser \
     --use_geopotential=True \
     --velocity_init_inflow_opt=FROM_SPEED_ANGLE_FLAGS \
     --wind_angle=0.0 --wind_speed=6.4573874 \
     --y_o_init=0.21 \
     --data_dump_prefix=<GCS_DIR> \
     --config_filepath=swirl-lm/swirl_lm/example/fire/fire.pbtxt \
     --target=<TPU>
   ```

   Note: `<GCS_DIR>` should be a path to a folder in an existing GCS
   bucket. It's OK if the folder doesn't yet exist - the solver will create
   it. But the bucket should exist before the solver runs. `<TPU>` should be the
   name of the TPU pod from the first step. Adjust cx, cy, cz and nx, ny, nz to
   match the number of TPUs you created in the first step.

   Tip: An introduction about the flags can be found in the header docstring of
   `swirl_lm/example/fire/fire_main.py`.

1. If you lose your ssh session, you can connect back to the same screen
   terminal by ssh'ing into the machine again and running `screen -rd`. You can
   also disconnect from ssh terminal by typing "Ctrl-a Ctrl-d" which will leave
   screen (and the simulation) running in the background and put you back to the
   original ssh terminal.

1. Don't forget to delete the TPU nodes when you are done.

   ```sh
   gcloud compute tpus execution-groups delete --zone=$ZONE $TPU
   ```

   Note: This deletes the machines and the disks, so if you want to restart
   the simulation you will need to clone and set up Swirl-LM again.
