# Streaming with ptychodus-stream

`ptychodus-stream` estimates the probe and object while a scan is acquired, publishes them as images for viewing, and saves each finished scan to a storage root that `ptychodus-webapp` browses and `ptychodus-reconstruct` reconstructs from. It runs as one [PvaPy](https://github.com/epics-base/pvaPy) area-detector consumer.

The first half of this page is the PV contract a beamline must meet; the second walks through a replay of the fly001 example scan with `scripts/replay_scan.py`, the reference producer for that contract.

## Installing

```sh
uv sync --extra stream --extra webapp
```

The `stream` extra installs pvapy; the `webapp` extra is only needed to browse the storage root.

## PV contract

Everything the processor reads and writes, stated for beamline scientists and controls engineers connecting real sources. Every name below that ptychodus defines is snake_case; names pvapy or areaDetector define (`uniqueId`, the NTNDArray fields) keep their own spelling.

### Detector

- An NTNDArray on `input_channel`, one 2-D frame per update, as areaDetector's PVA plugin publishes.
- `uniqueId` is the frame's **scan index**: the number the positioner gives the same point, allowing for `index_offset`. Indexes need not be contiguous or ordered, but must not repeat within a scan.
- One shape and dtype per scan. A frame of another shape or dtype than the scan's first is rejected and counted.
- Exactly one consumer receives every frame; `ptychodus-stream` always starts one.
- The frame a consumer connects on is discarded, as pvapy discards the value any monitor connects with; start `ptychodus-stream` before the scan.

### Positions

Positions arrive on `position_channel` over PVA, in one of two schemas chosen by `position_schema`. A point's position pairs with the frame whose `uniqueId` equals its index plus `index_offset`. Each axis is converted to metres by multiplying by its `scale_m`; a negative scale flips the axis.

#### `batched`

One structure with a field per configured axis, each axis

| Field | Type | Meaning |
| --- | --- | --- |
| `num_events` | `ulong` | Number of points in this batch |
| `event_ids` | `ulong[]` | Point indexes |
| `positions` | `double[]` | Values, in the axis's unit |

The fields are named by `x_axis.name`, `y_axis.name` and the optional `i0_name`. A batch may hold any number of points, and an axis may be left out of a batch.

#### `sgsocket`

One NTNDArray per point: its `uniqueId` is the point's index, and Float64 NDAttributes named by `x_axis.name`, `y_axis.name` and the optional `i0_name` carry the values. The array data is not read.

#### Rules for both schemas

- Positions may lag their frames by up to `settle_s` at the end of a scan; during a scan they may lag up to `unpaired_retention_s`.
- Axes may arrive at different rates and with different indexes: each is interpolated on its own at the frame indexes. Repeated indexes are averaged, gaps between indexes are interpolated linearly, and nothing is extrapolated.
- A frame joins its scan only once both x and y cover its index.

### I0

Optional incident-beam readings per point, on the axis `i0_name`, in any unit. They correct the totals for beam decay; `i0_photons_per_count` converts them to photons, and zero leaves them uncalibrated. I0 is used only when it covers every paired frame.

### Trigger

- A scalar PV, `trigger_pv`, over CA or PVA as `pv_protocol` says, which the processor can write.
- Write a nonzero value after the scan's last point. The processor writes 0 back at once, so the next scan's trigger cannot be lost.
- The value the PV holds when the processor connects is ignored.
- Everything arriving within `settle_s` of the trigger still belongs to the finishing scan; the next scan must start no sooner than `settle_s` after the trigger.

### Scan number and tomography angle

Optional scalar PVs `scan_number_pv` and `tomography_angle_pv` (in degrees). Write the new scan's values before its trigger: the processor monitors both and takes their values when the trigger arrives. The scan number labels the scan (`scan <n>`; otherwise the scan's start time), and the angle is recorded with it (otherwise `tomography_angle_deg`).

### Index rules

- Indexes must not repeat within a scan; repeats and indexes that go backwards are counted and logged.
- A counter that restarts each scan must restart on both the detector and the positioner, and the old scan's positions must all be sent before the new scan starts.

### Outputs

| Channel or path | Content |
| --- | --- |
| `<output_pv_prefix>:probe:amplitude`, `:probe:phase` | Float32 2-D NTNDArray of the first probe mode |
| `<output_pv_prefix>:object:amplitude`, `:object:phase` | Float32 2-D NTNDArray of the first object layer; it grows with the scan's extent |
| `<output_pv_prefix>:scan` | `{label, diffraction_uuid, product_uuid, num_frames_paired, num_frames_dropped, written}` for each finished scan |
| `<output_pv_prefix>:status` | The consumer's status, with the counters below |
| `<output_pv_prefix>:control` | `{"command": "configure", "args": "<json>"}` applies a partial configuration |
| `<output_directory>/diffraction/<uuid>/` | The scan's prepared patterns, `diffraction.h5`, with its manifest |
| `<output_directory>/product/<uuid>/` | The estimated probe, object and positions, `product.h5`, derived from that diffraction |

A `configure` names only the keys to change. It is validated first and rejected whole if invalid; a valid one discards the scan in progress and starts a new one, and `{}` is the reset. Changing a channel name needs a restart.

### Data handling

| Situation | What happens | Counter |
| --- | --- | --- |
| A frame's position has not arrived | It waits, up to `unpaired_retention_s` | `num_frames_pending` |
| A frame is never covered by positions | Dropped when it expires or at the end of the settle | `num_frames_dropped` |
| Frames between scans (free-running detector) | Never covered, so dropped as above | `num_frames_dropped` |
| A frame of another shape or dtype | Rejected | `num_frames_rejected` |
| A frame outside the total-count bounds | Filtered, as batch assembly does | `num_frames_filtered` |
| A repeated or decreasing index | Kept, and logged once per scan | `num_frames_duplicated`, `num_frames_regressed` |
| A mid-scan `configure` | The scan is discarded | `num_frames_discarded` |
| A trigger with no frames, or while settling | Acknowledged and ignored | none |
| An acknowledgement that fails | Logged; the scan still finishes | `num_trigger_failures` |

Further counters report the queue (`num_frames_queued`, `max_frames_queued`), the estimates (`num_estimates`, `last_estimate_duration_s`, `last_estimate_age_s`), storage (`num_scans_stored`), errors (`num_ingest_errors`, `num_worker_errors`) and thread health (`is_ingest_alive`, `is_worker_alive`).

### Bring-up checklist

1. Replay a recorded scan from the beamline with `scripts/replay_scan.py` and confirm it is stored with `num_frames_dropped` near zero.
2. Point the configuration at the real detector, positioner and trigger, and compare `<prefix>:scan` against the replay for one scan.
3. If `num_frames_dropped` is high, check `index_offset` (the detector's `uniqueId` often starts at 1) and that positions arrive within `settle_s` of the trigger.
4. If `max_frames_queued` keeps growing, ingest is slower than acquisition: reduce `crop_width_px`/`crop_height_px` or bin in the `pipeline`.

## Demonstration with fly001

The fly001 example scan (an APS Velociprobe fly scan) is not shipped with ptychodus; the paths below are placeholders for a local copy.

### Configure

Save as `fly001.yaml`:

```yaml
# Channels. The replay hosts every PV over PVA.
input_channel: fly001:image
position_channel: fly001:positions
trigger_pv: fly001:trigger
scan_number_pv: fly001:scan_number
output_pv_prefix: ptychodus
pv_protocol: pva

# Experiment
photon_energy_eV: 10000.0
detector_distance_m: 2.335
detector_pixel_size_m: 7.5e-05

# Preparation: a 128 px square around the beam centre
crop_center_x_px: 540
crop_center_y_px: 259
crop_width_px: 128
crop_height_px: 128

# Positions, sent in metres
position_schema: batched
x_axis: {name: x, scale_m: 1.0}
y_axis: {name: y, scale_m: 1.0}

# Estimation and timing
parallax_bin_size_px: 4
estimate_period_s: 5.0
settle_s: 2.0

# Storage
output_directory: /path/to/storage
campaign_label: fly001 demo
```

The other keys keep their defaults: maps (`bad_pixels_file`, `background_file`, `flat_field_file`, each `{file_path, file_type}`), `pipeline` steps, total-count bounds, `index_offset`, `i0_name` and `i0_photons_per_count`, the models (`probe_generator`, `object_generator` and their `*_options`), parallax thresholds, `unpaired_retention_s`, `num_frames_per_block`, `consumer_queue_size` and `queue_warning_frames`.

### Run

In one terminal, start the processor:

```sh
uv run ptychodus-stream --config fly001.yaml
```

In another, replay the scan:

```sh
uv run python scripts/replay_scan.py --config fly001.yaml \
    --diffraction-file /path/to/fly001/fly001_master.h5 --diffraction-file-type APS_Velociprobe \
    --position-file /path/to/fly001/fly001_pos.csv --position-file-type CSV
```

`--list-file-types` prints every reader name.

### Watch

The four image channels update every `estimate_period_s` and once more when the scan finishes; any areaDetector viewer that reads NTNDArrays shows them, or:

```sh
pvget ptychodus:probe:amplitude
pvget ptychodus:scan
```

The log reports the parallax defocus. fly001 records more frames than positions, so the frames past the last position are dropped at the end of the settle and counted in `num_frames_dropped`.

### Browse

```sh
PTYCHODUS_WEBAPP_STORAGE_ROOT=/path/to/storage uv run ptychodus-webapp serve
```

The scan appears under the `fly001 demo` campaign as a diffraction resource and the product derived from it.

### Reconstruct

```sh
uv run ptychodus-reconstruct --storage-root /path/to/storage --product <product_uuid> -o result
```

`<product_uuid>` is the one `ptychodus:scan` reported; with `PTYCHODUS_WEBAPP_STORAGE_ROOT` set, `--storage-root` can be left out.

### Variations

Each exercises one acquisition scenario; watch `pvget ptychodus:status` for the counters.

| Replay options | Scenario |
| --- | --- |
| `--position-delay-s 1` | Positions trail their frames; they still pair within the settle |
| `--repeat 3 --gap-s 3` | Back-to-back scans; each is stored, and no trigger is lost |
| `--repeat 2 --restart-indexes` | A counter restarting each scan |
| `--repeat 2 --free-run-s 30` | A free-running detector between scans; its frames are dropped |
| `position_schema: sgsocket` in the configuration | One event per point instead of batches |
| `x_axis: {name: x, scale_m: 1.0e-6}` | Positions sent in micrometres |

A mid-scan `configure` discards the scan in progress:

```sh
pvput ptychodus:control '{"command": "configure", "args": "{\"settle_s\": 3.0}"}'
```

## Memory and throughput

Only prepared frames are kept: cropped, binned by the pipeline, masked and filtered. A scan's frames are held until it is stored, alongside the next scan's while that happens; 10⁴ frames of 512×512 `uint16` are about 5 GB, and a crop or binning shrinks that in proportion. Frames waiting for positions are bounded by `unpaired_retention_s`.

Frames are queued as they arrive and prepared on a thread of their own, so a slow estimate or write never drops a frame; a backlog past `queue_warning_frames` is logged, and `max_frames_queued` records the worst so far. pvapy's own input queue, sized by `consumer_queue_size`, is the only place a frame can be lost, if it ever fills.
