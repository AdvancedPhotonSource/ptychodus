# Globus Compute Workflow

Perform steps 1-3 on the local and remote computers.

1. Install [miniforge](https://github.com/conda-forge/miniforge).

2. Make conda available in your current shell environment

   ```sh
   $ eval "$(~/miniforge3/bin/conda shell.bash hook)"
   ```

3. Create and activate the `ptychodus` conda environment

   ```sh
   $ conda create -c conda-forge -n ptychodus ptychodus-all
   $ conda activate ptychodus
   ```

4. Install a [Globus compute endpoint](https://globus-compute.readthedocs.io/en/stable/quickstart.html#deploying-an-endpoint) into the `ptychodus` environment on the remote computer and configure it. An example configuration file for ALCF Polaris is bundled with the Ptychodus source distribution.

   ```sh
   $ python -m pip install globus-compute-endpoint
   $ globus-compute-endpoint configure
   ```

5. Start the Globus compute endpoint

   ```sh
   $ globus-compute-endpoint start <ENDPOINT_NAME>
   ```

6. For data transfer, use a [guest collection](https://docs.globus.org/how-to/guest-collection-share-and-access) on a Globus Connect Server or use a [Globus Connect Personal endpoint](https://www.globus.org/globus-connect-personal) on your local computer.

7. On the local computer, launch the reconstruction tasks from the Globus panel.

8. On the remote computer, watch the queue (use qstat on Polaris) and Globus compute endpoint logs

   ```sh
   $ tail -f ~/.globus-compute/default/endpoint.log
   ```

9. When the demo is done, stop the Globus compute endpoint on the remote computer

   ```sh
   $ globus-compute-endpoint stop
   ```

## What a submission runs

A submission stages a directory for the selected product, transfers it to the compute collection, runs one command over it on the endpoint, and transfers the results back.

- A reconstruction stages `diffraction.h5`, `product.h5` and `reconstruct_options.yaml` — the selected library, algorithm and options — plus the trained model for a learned reconstructor, and runs `ptychodus-reconstruct` over them.
- Training stages the product's training data in the selected backend's format and `train_options.yaml`, and runs `ptychodus-train`.

Each writes its outputs, its options and its log into the output directory, with the command's standard output and error in `stdout.log` and `stderr.log`. Any library installed in the endpoint's environment can reconstruct.

## Example Globus Compute Endpoint Configuration

Here is an example Globus compute endpoint `config.yaml` for ALCF Polaris:

```{literalinclude} polaris.yaml
:language: yaml
```
