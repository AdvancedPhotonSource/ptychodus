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

7. On the local computer, launch the reconstruction tasks from the "Workflow" view.

8. On the remote computer, watch the queue (use qstat on Polaris) and Globus compute endpoint logs

   ```sh
   $ tail -f ~/.globus-compute/default/endpoint.log
   ```

9. When the demo is done, stop the Globus compute endpoint on the remote computer

   ```sh
   $ globus-compute-endpoint stop
   ```

## Example Globus Compute Endpoint Configuration

Here is an example Globus compute endpoint `config.yaml` for ALCF Polaris:

```{literalinclude} polaris.yaml
:language: yaml
```
