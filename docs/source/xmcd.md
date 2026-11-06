# XMCD Decomposition

X-ray magnetic circular dichroism makes a magnetic sample's transmission depend on the helicity of the beam. Ptychodus separates two reconstructions of the same region, one measured with right-circularly polarized (RCP) light and one with left-circularly polarized (LCP) light, into a helicity-independent structural object and a magnetic factor. The decomposition models the two complex transmissions as

```text
O+ (RCP) = O_struct * M
O- (LCP) = O_struct / M
```

where `M = |M| exp(i alpha)` carries both the magnetic absorption and phase contrast. The magnetic factor is recovered first, taking its phase on the principal branch, where a magnetic factor near unity lies; the structural object is then derived from it, so that `O_struct * M` reproduces the RCP object and `O_struct / M` the LCP object exactly.

## What the outputs mean

The LCP object is aligned to the RCP object with sub-pixel precision, both are trimmed to their common shape, and the LCP reconstruction's scale, phase offset and phase ramp are standardized against the RCP one. Each reconstruction carries those ambiguities independently, so the standardization is needed — but it cannot tell them from the magnetic signal:

- **Magnetic contrast is relative to the field of view.** The weighted mean magnetic phase, any linear trend in it, and the weighted mean of `|M|^2` are absorbed into the standardization. A uniform magnetization is not recovered; domains of opposite sign are.
- **Both outputs are on the RCP product's frame.** Each is the RCP product — its probe, probe positions and metadata — with the object replaced, named `<rcp>-structural` and `<rcp>-magnetic`, with a comment naming both inputs, no polarization and no loss history.
- **The magnetic product's object is the factor `M`**, not a transmission. It is a product so it keeps its pixel size and position on the sample, but it is not a starting point for a reconstruction.

Both reconstructions must share an object pixel size. A magnetic phase beyond ±π/2 wraps by π, and the structural object then changes sign with it.

## In the GUI

On the **Object** panel, choose **Analyze** → **XMCD...**, then pick the left and right circular products. The dialog shows the structural and magnetic objects; **Add to Products** adds both to the product list, where they are viewed, compared and saved like any other product.

## From the command line

`ptychodus-xmcd` decomposes two product files and writes two:

```sh
$ ptychodus-xmcd --lcp lcp/product.h5 --rcp rcp/product.h5 \
    --structural xmcd/structural.h5 --magnetic xmcd/magnetic.h5
```

Inputs and outputs are ptychodus product HDF5 files, such as the `product.h5` that `ptychodus-reconstruct` writes. The outputs must differ from each other and from both inputs. If a file's recorded polarization contradicts the option it was passed as, the command warns that the inputs may be swapped: swapping them inverts the magnetic contrast. `--dry-run` reads and checks the inputs without decomposing.

The alignment is tuned with the same options the decomposition takes:

| Option | Default | Meaning |
| --- | --- | --- |
| `--upsample-factor` | `100` | sub-pixel precision of the alignment, as 1/N pixel |
| `--registration-quantity` | `complex` | quantity correlated to align the objects: `complex`, `variation`, `phasor`, `phase` or `amplitude` |
| `--high-pass-sigma` | `0` | Gaussian background sigma in pixels removed before alignment; `0` disables it |
| `--epsilon` | `1e-12` | added to the LCP object before dividing, to keep the magnetic amplitude finite where it is zero |

`complex` is the only exact registration quantity, but it is sensitive to a phase ramp between the two reconstructions; `variation` is insensitive to one, at the cost of a small sub-pixel bias.

## How it works

From the aligned, standardized pair, the magnetic factor is recovered first and the structural object derived from it:

```text
alpha       = 0.5 * angle(O+ * conj(O-))
M           = sqrt(|O+ / (O- + epsilon)|) * exp(i alpha)
O_struct    = sqrt(|O+ * O-|) * exp(i angle(O- * exp(i alpha)))
```

The magnetic phase comes from the cross product `O+ * conj(O-)` rather than from the ratio `O+ / O-`, because the cross product involves no division and stays well conditioned where `|O-|` is small. The magnetic amplitude still needs the ratio, so `epsilon` is added to the LCP denominator. That only guards against exact zeros: the magnetic amplitude near such pixels stays noisy.

The model fixes `(O_struct, M)` only up to a joint sign, since `(-O_struct, -M)` reproduces both measurements. Taking `alpha` on the principal branch `(-π/2, π/2]` picks the `M` near the positive real axis, and deriving the structural phase from `alpha` keeps the pair consistent. Halving the structural and magnetic angles independently would instead flip the sign of `O_struct` alone wherever its phase crossed ±π/2.
