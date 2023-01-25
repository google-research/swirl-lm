# Swirl-LM Demo on Cloud TPU

This is an example of a stand-alone python script that uses the Swirl-LM
library. It demonstrates how to run a Taylor-Green Vortex Flow on Google
Cloud TPUs.

1. Check / set project:

   Check if your GC project is set on your local machine:

   ```sh
   gcloud config list
   ```

   If the output does not contain the correct project, then set it with:

   ```sh
   gcloud config set project <PROJECT>
   ```

1. Create the TPU nodes and VM:

   ```sh
   TPU=swirl-lm-demo
   ZONE=europe-west4-a
   gcloud compute tpus execution-groups create \
     --zone="$ZONE" --name="$TPU" --accelerator-type=v3-32 \
     --tf-version=2.9.1
   ```

   This step creates a slice of 32 TPUs (v3) and a GCE VM. The TPU hosts and
   the VM are automatically configured to be able to communicate. The VM image
   includes TensorFlow.

   See https://cloud.google.com/tpu/docs/regions-zones for regions and TPU
   configurations. For most up-to-date info, use:

   ```sh
   gcloud compute tpus accelerator-types list --zone="$ZONE"
   ```

   Note: The VM and the TPU nodes have the same name. The name refers to the VM
   normally, and to the TPU cluster when used in the TPU APIs.

   Note: You might need to enable the TPU API for your project if it's not
   already enabled.

1. SSH into the VM:

   The previous command (execution-groups create) may automatically ssh into
   the VM. If not, ssh into the VM:

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

1. (VM) Run the solver.

   ```sh
    python3 swirl-lm/swirl_lm/example/tgv/main.py \
      --data_dump_prefix=gs://<GCS_DIR>/data/tgv \
      --data_load_prefix=gs://<GCS_DIR>/data/tgv \
      --config_filepath=swirl-lm/swirl_lm/example/tgv/tgv_3d.textpb \
      --cx=2 --cy=2 --cz=8 \
      --num_steps=2000 --nx=128 --ny=128 --nz=6 --kernel_size=16 \
      --halo_width=2 --lx=6.28 --ly=6.28 --lz=6.28 --num_boundary_points=0 \
      --dt=2e-3 --u_mag=1.0 --p_ref=0.0 --rho_ref=1.0
      --target=<TPU> \
      --output_fn_template=tgv_{var}.png
   ```

   Note: `<GCS_DIR>` should be a path to a folder in an existing GCS
   bucket. It's OK if the folder doesn't yet exist - the solver will create
   it. But the bucket should exist before the solver runs.

1. (VM) Check that the output file has been created.

   ```sh
   ls -l tgv_*.png
   ```

   Run the remaining commands are on your local machine and not on the VM.

1. Copy the output out of the VM to view it locally.

   ```sh
   gcloud compute scp --zone=$ZONE $TPU:tgv_*.png /tmp
   ```

   Note: Alternatively, you can set `--output_fn_template` to point to a
   location in a GCS bucket and access the files through the GCS browser.

1. Delete the TPU nodes and VM.

   ```sh
   gcloud compute tpus execution-groups delete --zone=$ZONE $TPU
   ```

   Note: This deletes both the TPUs and the VM. Deleting the VM also deletes
   its disk by default, so you will lose the cloned repo, etc. If you plan to
   re-run, you can delete only the TPUs by passing `--tpu-only` to the
   command above; and later create only the TPUs by again passing the same
   flag.
