"""
process_roi.py — single-function IMC ROI processing pipeline
Builds up step by step. Block 1: setup + load + normalize.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import tifffile
import matplotlib.pyplot as plt
from readimc import MCDFile
from cellpose import models
from skimage.color import label2rgb
from skimage.segmentation import mark_boundaries
from skimage import filters, morphology, measure
from scipy import ndimage as ndi
from skimage.measure import regionprops_table



def plot_marker_in_compartment(result, marker_name, compartment="dermis",
                                cmap="hot", percentile=(1, 99)):
    """Show a marker's expression restricted to a tissue compartment."""
    idx = next(i for i, lbl in enumerate(result["channel_labels"])
               if marker_name in lbl)
    img = result["images_norm"][idx]
    mask = result[f"{compartment[:4]}_mask"]  # epi_mask or derm_mask

    masked = np.where(mask, img, np.nan)
    lo, hi = np.nanpercentile(masked, percentile)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(masked, cmap=cmap, vmin=lo, vmax=hi)
    ax.set_title(f"{result['roi_tag']}  {marker_name}  ({compartment})")
    ax.axis("off")
    return fig


def process_roi(
    roi_tag: str,
    mcd_dir: str | Path = "./",
    output_dir: str | Path = "./roi_output",
    dna_channel: str = "DNA1",
    ck_channel: str = "CK",
    arcsinh_cofactor: float = 1.0,
):
    """
    Process a single IMC ROI end-to-end.

    Parameters
    ----------
    roi_tag : str
        ROI identifier like 'IMC1_001' → file 'IMC1.mcd', acquisition #1
    mcd_dir : Path
        Directory containing the .mcd files
    output_dir : Path
        Where to save mask TIFFs, overlay PNGs, and the per-cell CSV
    dna_channel : str
        Substring to match the nuclear channel (default 'DNA1')
    ck_channel : str
        Substring to match the cytokeratin channel (default 'CK')
    arcsinh_cofactor : float
        Cofactor for arcsinh transform. IMC standard = 1.0

    Returns
    -------
    df : pd.DataFrame
        One row per cell with morphology, compartment, and arcsinh intensities
    """

    # ─── BLOCK 1: setup, parse tag, load acquisition, normalize ──────────────
    mcd_dir    = Path(mcd_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # parse 'IMC1_001' → file_stem='IMC1', acq_id=1
    file_stem, acq_id_str = roi_tag.split("_")
    acq_id  = int(acq_id_str)
    mcd_path = mcd_dir / f"{file_stem}.mcd"

    if not mcd_path.exists():
        raise FileNotFoundError(f"MCD file not found: {mcd_path}")

    print(f"[{roi_tag}]  File: {mcd_path.name}   Acquisition: {acq_id}")

    # locate the acquisition: match by description containing the ID
    # OR fall back to positional index (acq_id - 1)
    with MCDFile(mcd_path) as mcd:
        target_acq = None
        for slide in mcd.slides:
            for idx, acq in enumerate(slide.acquisitions):
                desc = acq.description or ""
                if str(acq_id) in desc or idx == acq_id - 1:
                    target_acq = acq
                    break
            if target_acq:
                break

        if target_acq is None:
            raise ValueError(f"Acquisition {acq_id} not found in {mcd_path.name}")

        # extract image stack (C, H, W) and metadata
        images_raw     = mcd.read_acquisition(target_acq)              # float32
        channel_labels = list(target_acq.channel_labels)
        channel_metals = list(target_acq.channel_metals)
        channel_masses = list(target_acq.channel_masses)
        pixel_um       = target_acq.pixel_size_x_um

    print(f"[{roi_tag}]  Shape: {images_raw.shape}   pixel size: {pixel_um:.3f} µm")

    # arcsinh normalization (IMC standard, cofactor=1)
    # variance-stabilizes counts, compresses dynamic range, keeps zeros at 0
    images_norm = np.arcsinh(images_raw / arcsinh_cofactor)

    print(f"[{roi_tag}]  Normalized via arcsinh (cofactor={arcsinh_cofactor})")
    print(f"[{roi_tag}]  Channels available:")
    for i, (m, mass, lbl) in enumerate(zip(channel_metals, channel_masses, channel_labels)):
        print(f"            [{i:02d}] {m}{mass}  {lbl}")

    '''# for use in next blocks — return these temporarily so you can inspect
                return {
                    "roi_tag":        roi_tag,
                    "images_norm":    images_norm,
                    "channel_labels": channel_labels,
                    "channel_metals": channel_metals,
                    "channel_masses": channel_masses,
                    "pixel_um":       pixel_um,
                    "output_dir":     output_dir,
                }'''

    # ─── BLOCK 2: cell segmentation on DNA1 with Cellpose ────────────────────

    # find DNA channel index
    try:
        dna_idx = next(i for i, lbl in enumerate(channel_labels)
                       if dna_channel in lbl)
    except StopIteration:
        raise ValueError(f"DNA channel '{dna_channel}' not found. "
                         f"Available: {channel_labels}")

    dna_img = images_norm[dna_idx]
    print(f"[{roi_tag}]  Segmenting cells on channel {dna_idx}: "
          f"{channel_labels[dna_idx]}")

    # Cellpose-SAM model (cellpose 4.x default)
    # gpu=True will use MPS on Apple Silicon; falls back to CPU automatically
    model = models.CellposeModel(gpu=True)

    dna_raw   = images_raw[dna_idx]
    lo, hi    = np.percentile(dna_raw, [1, 99.5])
    dna_for_cp = np.clip((dna_raw - lo) / (hi - lo + 1e-9), 0, 1)

    cell_masks, flows, styles = model.eval(
        dna_for_cp,
        diameter=12,             # ~10 µm nuclei at 1 µm/px — tune per tissue
        flow_threshold=0.4,
        cellprob_threshold=-2,
        normalize=True,          # cellpose's internal percentile stretch
        min_size=10,             # discard tiny segments
    )
    n_cells = int(cell_masks.max())
    print(f"[{roi_tag}]  Segmented {n_cells} cells")

    # save raw label TIFF (for Fiji / downstream tools)
    tifffile.imwrite(
        output_dir / f"{roi_tag}_cell_masks.tiff",
        cell_masks.astype(np.int32)
    )

    # save colorized overlay PNG (for human eyes)
    # normalize DNA to 0-1 for display
    lo, hi = np.percentile(dna_img, [1, 99])
    dna_disp = np.clip((dna_img - lo) / (hi - lo + 1e-9), 0, 1)
    dna_rgb  = np.stack([dna_disp] * 3, axis=-1)

    # boundaries overlaid on DNA
    boundaries = mark_boundaries(dna_rgb, cell_masks,
                                  color=(1.0, 0.4, 0.0), mode="thick")

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), facecolor="black")
    axes[0].imshow(label2rgb(cell_masks, image=dna_rgb,
                              bg_label=0, alpha=0.5))
    axes[0].set_title(f"{n_cells} segmented cells",
                       color="white", fontsize=11)
    axes[1].imshow(boundaries)
    axes[1].set_title("Cell boundaries on DNA",
                       color="white", fontsize=11)
    for ax in axes:
        ax.axis("off")
    plt.suptitle(f"{roi_tag} — Cellpose segmentation",
                 color="white", fontsize=13)
    plt.tight_layout()
    fig.savefig(output_dir / f"{roi_tag}_cell_segmentation.png",
                dpi=150, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    print(f"[{roi_tag}]  Saved cell segmentation overlay")

    

    # ─── BLOCK 3: epidermis / dermis compartment segmentation ────────────────

    # find CK channel(s) — if multiple keratins, max-project them
    ck_indices = [i for i, lbl in enumerate(channel_labels)
                  if ck_channel.lower() in lbl.lower()]
    if not ck_indices:
        raise ValueError(f"No CK channel found matching '{ck_channel}'. "
                         f"Available: {channel_labels}")

    if len(ck_indices) == 1:
        ck_img = images_norm[ck_indices[0]]
        print(f"[{roi_tag}]  Using CK channel: "
              f"{channel_labels[ck_indices[0]]}")
    else:
        ck_img = images_norm[ck_indices].max(axis=0)
        print(f"[{roi_tag}]  Max-projecting {len(ck_indices)} CK channels: "
              f"{[channel_labels[i] for i in ck_indices]}")

    # ─── 3a. tissue mask (more permissive — captures sparse dermis) ──────────
    # Use BOTH DNA and CK signal — dermis has low DNA but the sum of all
    # markers is a better tissue proxy than DNA alone
    # Sum a few "everywhere" channels: DNA + CK + total signal
    total_signal = images_norm.sum(axis=0)               # sum over all channels
    total_blur   = filters.gaussian(total_signal, sigma=4)

    # use a more permissive threshold than Otsu — triangle works better
    # for skewed distributions where tissue is much larger than background
    tissue_thresh = filters.threshold_triangle(total_blur)
    tissue_mask   = total_blur > tissue_thresh

    # aggressive closing to bridge gaps in sparse dermis
    tissue_mask   = morphology.closing(tissue_mask, morphology.disk(20))
    tissue_mask   = ndi.binary_fill_holes(tissue_mask)
    tissue_mask   = morphology.remove_small_objects(tissue_mask, min_size=5000)

    # gentle expansion to capture the very edge of dermis
    tissue_mask   = morphology.binary_dilation(tissue_mask, morphology.disk(5))

    # ─── 3b. epidermis = CK-high regions within tissue ───────────────────────
    ck_blur = filters.gaussian(ck_img, sigma=2)
    # threshold INSIDE tissue only — global Otsu would split tissue vs slide
    if tissue_mask.any():
        epi_thresh = filters.threshold_otsu(ck_blur[tissue_mask])
    else:
        epi_thresh = filters.threshold_otsu(ck_blur)
    epi_mask = (ck_blur > epi_thresh) & tissue_mask

    # connect broken-up keratinocyte regions
    epi_mask = morphology.closing(epi_mask, morphology.disk(5))

    # drop tiny CK+ specks (occasional immune cells weakly express CK, debris)
    min_epi_um2 = 2000                                   # ~45×45 µm patch
    min_epi_px  = int(min_epi_um2 / (pixel_um ** 2))
    epi_mask    = morphology.remove_small_objects(epi_mask, min_size=min_epi_px)

    # capture basal cells right at the basement membrane
    dilate_um   = 5
    dilate_px   = max(1, int(dilate_um / pixel_um))
    epi_mask    = morphology.binary_dilation(epi_mask,
                                              morphology.disk(dilate_px))
    epi_mask    = epi_mask & tissue_mask                 # never extend outside tissue

    # ─── 3c. dermis = tissue minus epidermis ─────────────────────────────────
    derm_mask = tissue_mask & ~epi_mask

    # report compartment areas
    epi_area_um2  = float(epi_mask.sum()  * pixel_um ** 2)
    derm_area_um2 = float(derm_mask.sum() * pixel_um ** 2)
    print(f"[{roi_tag}]  Epidermis: {epi_area_um2/1e6:.3f} mm²   "
          f"Dermis: {derm_area_um2/1e6:.3f} mm²")

    # ─── 3d. save compartment overlay PNG ────────────────────────────────────
    # CK background in grayscale, with red=epi and green=dermis overlay
    lo, hi  = np.percentile(ck_img, [1, 99])
    ck_disp = np.clip((ck_img - lo) / (hi - lo + 1e-9), 0, 1)

    overlay = np.zeros((*ck_img.shape, 3))
    overlay[..., 0] = epi_mask  * 0.7                    # red = epidermis
    overlay[..., 1] = derm_mask * 0.4                    # green = dermis

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor="black")
    axes[0].imshow(ck_disp, cmap="hot")
    axes[0].set_title(f"CK signal", color="white", fontsize=11)
    axes[1].imshow(tissue_mask, cmap="gray")
    axes[1].set_title("Tissue mask", color="white", fontsize=11)
    axes[2].imshow(ck_disp, cmap="gray")
    axes[2].imshow(overlay, alpha=0.45)
    axes[2].set_title("Epidermis (red) / Dermis (green)",
                      color="white", fontsize=11)
    for ax in axes:
        ax.axis("off")
    plt.suptitle(f"{roi_tag} — Tissue compartment segmentation",
                 color="white", fontsize=13)
    plt.tight_layout()
    fig.savefig(output_dir / f"{roi_tag}_compartments.png",
                dpi=150, bbox_inches="tight", facecolor="black")
    plt.close(fig)

    # save the raw masks as TIFFs too — useful for downstream / Fiji
    tifffile.imwrite(output_dir / f"{roi_tag}_tissue_mask.tiff",
                     tissue_mask.astype(np.uint8))
    tifffile.imwrite(output_dir / f"{roi_tag}_epidermis_mask.tiff",
                     epi_mask.astype(np.uint8))
    tifffile.imwrite(output_dir / f"{roi_tag}_dermis_mask.tiff",
                     derm_mask.astype(np.uint8))
    print(f"[{roi_tag}]  Saved compartment overlay + mask TIFFs")




    # ─── BLOCK 4: per-cell feature extraction + compartment assignment ───────
    
    # 4a. morphology features
    morph = pd.DataFrame(regionprops_table(
        cell_masks,
        properties=["label", "area", "eccentricity", "perimeter",
                    "solidity", "centroid"]
    ))
    morph.rename(columns={"centroid-0": "centroid_y",
                          "centroid-1": "centroid_x"}, inplace=True)
    morph["area_um2"] = morph["area"] * (pixel_um ** 2)

    # 4b. mean arcsinh intensity per channel per cell
    intensity_data = {"label": morph["label"].values}
    for i, lbl in enumerate(channel_labels):
        col_name = f"{channel_metals[i]}{channel_masses[i]}_{lbl}"
        props = regionprops_table(
            cell_masks,
            intensity_image=images_norm[i],
            properties=["label", "intensity_mean"]
        )
        intensity_data[col_name] = pd.DataFrame(props)["intensity_mean"].values
    intensities = pd.DataFrame(intensity_data)

    # 4c. assign each cell to a compartment by its centroid
    def assign_compartment(row):
        y, x = int(row.centroid_y), int(row.centroid_x)
        # clamp to image bounds (centroids should always be in-bounds, but safe)
        y = min(max(y, 0), epi_mask.shape[0] - 1)
        x = min(max(x, 0), epi_mask.shape[1] - 1)
        if epi_mask[y, x]:
            return "epidermis"
        if derm_mask[y, x]:
            return "dermis"
        return "other"            # cell outside both compartments (rare edge)

    morph["compartment"] = morph.apply(assign_compartment, axis=1)

    # 4d. assemble final dataframe
    df = morph.merge(intensities, on="label")
    df.insert(0, "roi", roi_tag)                          # tag every cell with ROI

    # report compartment breakdown
    comp_counts = df["compartment"].value_counts().to_dict()
    print(f"[{roi_tag}]  Cells by compartment: {comp_counts}")

    # save the per-cell table
    df.to_csv(output_dir / f"{roi_tag}_cells.csv", index=False)
    print(f"[{roi_tag}]  Saved cell table: "
          f"{len(df)} cells × {len(df.columns)} features")

    # ─── final return: complete dictionary ───────────────────────────────────
    return {
        "roi_tag":        roi_tag,
        "df":             df,
        "images_norm":    images_norm,
        "images_raw":     images_raw,
        "channel_labels": channel_labels,
        "channel_metals": channel_metals,
        "channel_masses": channel_masses,
        "pixel_um":       pixel_um,
        "cell_masks":     cell_masks,
        "n_cells":        n_cells,
        "tissue_mask":    tissue_mask,
        "epi_mask":       epi_mask,
        "derm_mask":      derm_mask,
        "epi_area_um2":   epi_area_um2,
        "derm_area_um2":  derm_area_um2,
        "output_dir":     output_dir,
    }


if __name__ == "__main__":

    PN_L_ROI = ['IMC2_002', 'IMC3_002', 'IMC3_004', 'IMC4_005', 'IMC5_005', 'IMC6_004', 'IMC1_003', 'IMC1_004']
    PN_NL_ROI = ['IMC1_001', 'IMC1_002', 'IMC2_002', 'IMC3_001', 'IMC3_003', 'IMC4_004', 'IMC5_004', 'IMC6_003']
    HC_ROI = ['IMC2_005', 'IMC3_005', 'IMC4_003', 'IMC5_001', 'IMC6_001']

    PN_L_ROI = ['IMC7_003', 'IMC8_001', 'IMC9_004', 'IMC9_005', 'IMC10_004', 'IMC10_005', 'IMC11_003', 'IMC11_004', 'IMC12_003', 'IMC12_004', 'IMC13_001', 'IMC13_003', 'IMC14_003', 'IMC14_004', 'IMC15_003', 'IMC15_004', 'IMC16_004']
    PN_NL_ROI = ['IMC8_005','IMC9_002', 'IMC9_003', 'IMC10_003', 'IMC11_001', 'IMC11_002', 'IMC12_002', 'IMC12_005', 'IMC13_002', 'IMC13_004', 'IMC14_002', 'IMC15_001', 'IMC15_002', 'IMC16_001']
    HC_ROI = ['IMC7_001', 'IMC8_003', 'IMC14_001', 'IMC15_005']
    

    for roi in PN_L_ROI+PN_NL_ROI+HC_ROI:
        res = process_roi(roi, mcd_dir="./raw_data/", output_dir=f"./roi_output/{roi}")
        print(f"\n=== Summary for {res['roi_tag']} ===")
        print(f"Total cells: {res['n_cells']}")
        print(f"By compartment: {res['df']['compartment'].value_counts().to_dict()}")
        print(f"Epidermis area: {res['epi_area_um2']/1e6:.3f} mm²")
        print(f"Dermis area:    {res['derm_area_um2']/1e6:.3f} mm²")
        print(f"\nDataFrame preview:")
        print(res['df'].head())