"""
3D Renderer
============
Segmentation NIfTI → photorealistic VTK PNG renders
"""

import os
import numpy as np
import SimpleITK as sitk
import vtk
from vtk.util import numpy_support
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Patch


LABEL_CONFIG = {
    1: {"name": "Whole Gland",     "color": (0.90, 0.68, 0.48), "opacity": 0.35, "smooth": 45},
    2: {"name": "Transition Zone", "color": (0.78, 0.18, 0.14), "opacity": 0.92, "smooth": 60},
    3: {"name": "Peripheral Zone", "color": (0.95, 0.78, 0.40), "opacity": 0.80, "smooth": 45},
}


def render_3d(seg_path, output_dir):
    """
    Load segmentation, build 3D surfaces, render 4 views + composite.
    Returns (view_list, stats_list)
    """
    # ── Load ─────────────────────────────────────────────
    img = sitk.ReadImage(seg_path)
    arr = np.round(sitk.GetArrayFromImage(img)).astype(np.uint8)
    sp  = img.GetSpacing()

    print(f"[Render] Shape={arr.shape} Spacing={sp} Labels={np.unique(arr)}")

    if np.sum(arr > 0) == 0:
        raise RuntimeError(
            "Segmentation is empty. "
            "The model may not have found the prostate — "
            "check that input is T2-weighted MRI."
        )

    # ── Clean labels before meshing ───────────────────────
    # Remove tiny disconnected islands and fill internal holes per label.
    cleaned = np.zeros_like(arr, dtype=np.uint8)
    for label in np.unique(arr):
        if label == 0:
            continue
        mask = arr == label
        if int(np.sum(mask)) < 64:
            continue
        cleaned[_cleanup_label_mask(mask)] = label
    arr = cleaned

    if np.sum(arr > 0) == 0:
        raise RuntimeError("Segmentation became empty after cleanup.")

    # ── Crop tightly ──────────────────────────────────────
    zs = np.where(np.sum(arr > 0, axis=(1, 2)) > 0)[0]
    ys = np.where(np.sum(arr > 0, axis=(0, 2)) > 0)[0]
    xs = np.where(np.sum(arr > 0, axis=(0, 1)) > 0)[0]
    p  = 4
    arr = arr[max(0, zs[0]-p):zs[-1]+p+1,
               max(0, ys[0]-p):ys[-1]+p+1,
               max(0, xs[0]-p):xs[-1]+p+1]

    # ── Resample to 1mm isotropic ─────────────────────────
    arr_img = sitk.GetImageFromArray(arr.astype(np.float32))
    arr_img.SetSpacing(sp)
    new_sz = [int(round(arr.shape[2]*sp[0])),
              int(round(arr.shape[1]*sp[1])),
              int(round(arr.shape[0]*sp[2]))]
    res = sitk.ResampleImageFilter()
    res.SetOutputSpacing([1.0, 1.0, 1.0])
    res.SetSize(new_sz)
    res.SetOutputDirection(arr_img.GetDirection())
    res.SetOutputOrigin(arr_img.GetOrigin())
    res.SetTransform(sitk.Transform())
    res.SetInterpolator(sitk.sitkNearestNeighbor)
    arr = np.round(sitk.GetArrayFromImage(res.Execute(arr_img))).astype(np.uint8)
    print(f"[Render] Isotropic: {arr.shape}")

    # ── VTK image ─────────────────────────────────────────
    arr_vtk = np.ascontiguousarray(arr.transpose(2, 1, 0))
    v = numpy_support.numpy_to_vtk(
        arr_vtk.flatten(), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR
    )
    image = vtk.vtkImageData()
    image.SetDimensions(arr_vtk.shape)
    image.SetSpacing(1.0, 1.0, 1.0)
    image.GetPointData().SetScalars(v)

    # ── Build surfaces ────────────────────────────────────
    actors   = []
    rendered = {}

    for label, cfg in LABEL_CONFIG.items():
        count = int(np.sum(arr == label))
        if count < 100:
            continue
        print(f"[Render] Building {cfg['name']} ({count:,} voxels)...")

        thresh = vtk.vtkImageThreshold()
        thresh.SetInputData(image)
        thresh.ThresholdBetween(label-0.5, label+0.5)
        thresh.SetInValue(255); thresh.SetOutValue(0); thresh.Update()

        gauss = vtk.vtkImageGaussianSmooth()
        gauss.SetInputConnection(thresh.GetOutputPort())
        gauss.SetStandardDeviations(1.5, 1.5, 1.5)
        gauss.SetRadiusFactors(3, 3, 3); gauss.Update()

        mc = vtk.vtkMarchingCubes()
        mc.SetInputConnection(gauss.GetOutputPort())
        mc.SetValue(0, 100); mc.Update()

        pts = mc.GetOutput().GetNumberOfPoints()
        print(f"  {pts:,} surface points")
        if pts == 0:
            continue

        conn = vtk.vtkConnectivityFilter()
        conn.SetInputConnection(mc.GetOutputPort())
        conn.SetExtractionModeToLargestRegion()
        conn.Update()

        sm = vtk.vtkWindowedSincPolyDataFilter()
        sm.SetInputConnection(conn.GetOutputPort())
        sm.SetNumberOfIterations(cfg["smooth"])
        sm.BoundarySmoothingOff()
        sm.FeatureEdgeSmoothingOff()
        sm.SetPassBand(0.08)
        sm.NonManifoldSmoothingOn()
        sm.NormalizeCoordinatesOn()
        sm.Update()

        nm = vtk.vtkPolyDataNormals()
        nm.SetInputConnection(sm.GetOutputPort())
        nm.ComputePointNormalsOn(); nm.SplittingOff()
        nm.ConsistencyOn(); nm.AutoOrientNormalsOn(); nm.Update()

        mp = vtk.vtkPolyDataMapper()
        mp.SetInputConnection(nm.GetOutputPort()); mp.ScalarVisibilityOff()

        ac = vtk.vtkActor(); ac.SetMapper(mp)
        pr = ac.GetProperty()
        pr.SetColor(*cfg["color"]); pr.SetOpacity(cfg["opacity"])
        pr.SetAmbient(0.12); pr.SetDiffuse(0.72)
        pr.SetSpecular(0.45); pr.SetSpecularPower(45)
        pr.SetInterpolationToPhong()

        actors.append(ac)
        rendered[label] = cfg

    if not actors:
        raise RuntimeError("No 3D surfaces could be built from segmentation")

    # ── Render views ──────────────────────────────────────
    views_config = [
        ("anterior",  20,   0, "Anterior View"),
        ("lateral",   20,  90, "Left Lateral View"),
        ("oblique",   35,  45, "Oblique View"),
        ("superior",  85,   0, "Superior View"),
    ]

    view_results = []
    for name, elev, azim, title in views_config:
        out_path = os.path.join(output_dir, f"view_{name}.png")
        _render_single(actors, rendered, out_path, elev, azim, title)
        view_results.append({"name": title, "file": f"view_{name}.png"})

    # ── Composite ─────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(22, 18))
    fig.patch.set_facecolor('#04040f')
    fig.suptitle("Prostate MRI — 3D Anatomical Segmentation",
                 color='white', fontsize=22, fontweight='bold', y=0.99)

    for ax, vr in zip(axes.flat, view_results):
        try:
            ax.imshow(mpimg.imread(os.path.join(output_dir, vr["file"])))
            ax.set_title(vr["name"], color='white', fontsize=14, fontweight='bold')
        except Exception as e:
            ax.text(0.5, 0.5, str(e), transform=ax.transAxes,
                    color='white', ha='center')
        ax.axis('off')

    legend = [
        Patch(facecolor=cfg["color"], label=cfg["name"])
        for label, cfg in rendered.items()
    ]
    fig.legend(handles=legend, loc='lower center', ncol=3,
               facecolor='#1a1a2e', labelcolor='white',
               fontsize=13, bbox_to_anchor=(0.5, 0.003))
    plt.tight_layout(rect=[0, 0.04, 1, 0.97])
    comp_path = os.path.join(output_dir, "composite.png")
    plt.savefig(comp_path, dpi=150, bbox_inches='tight', facecolor='#04040f')
    plt.close()
    print(f"[Render] Composite saved: {comp_path}")

    # ── Stats ─────────────────────────────────────────────
    stats = []
    for label, cfg in rendered.items():
        vox = int(np.sum(arr == label))
        cc  = round(vox / 1000.0, 1)  # 1mm^3 voxels → cc
        stats.append({
            "label":     label,
            "name":      cfg["name"],
            "voxels":    vox,
            "volume_cc": cc,
        })
        print(f"[Render] {cfg['name']}: {cc}cc")

    return view_results, stats


def _cleanup_label_mask(mask):
    """Keep largest connected component and fill interior holes."""
    mask_img = sitk.GetImageFromArray(mask.astype(np.uint8))
    cc = sitk.ConnectedComponent(mask_img)
    relabeled = sitk.RelabelComponent(cc, sortByObjectSize=True)
    largest = sitk.Equal(relabeled, 1)
    filled = sitk.BinaryFillhole(largest)
    closed = sitk.BinaryMorphologicalClosing(filled, [1, 1, 1])
    return sitk.GetArrayFromImage(closed).astype(bool)


def _render_single(actors, rendered, out_path, elev, azim, title):
    ren = vtk.vtkRenderer()
    ren.SetBackground(0.04, 0.04, 0.10)
    ren.GradientBackgroundOn()
    ren.SetBackground2(0.01, 0.01, 0.05)
    for a in actors:
        ren.AddActor(a)

    for pos, intensity, col in [
        ((300,  300,  400),  1.00, (1.00, 0.95, 0.90)),
        ((-200, 100,  100),  0.50, (0.70, 0.80, 1.00)),
        ((0,   -300, -100),  0.30, (1.00, 0.85, 0.70)),
        ((0,    0,   -400),  0.20, (0.85, 0.85, 1.00)),
    ]:
        l = vtk.vtkLight()
        l.SetLightTypeToSceneLight()
        l.SetPosition(*pos); l.SetFocalPoint(0, 0, 0)
        l.SetIntensity(intensity); l.SetColor(*col)
        ren.AddLight(l)

    # Legend
    y = 0.92
    for label, cfg in rendered.items():
        t = vtk.vtkTextActor()
        t.SetInput(f"  {cfg['name']}")
        t.GetTextProperty().SetFontSize(18)
        t.GetTextProperty().SetColor(*cfg["color"])
        t.GetTextProperty().BoldOn()
        t.GetPositionCoordinate().SetCoordinateSystemToNormalizedDisplay()
        t.SetPosition(0.02, y)
        ren.AddViewProp(t)
        y -= 0.07

    tt = vtk.vtkTextActor()
    tt.SetInput(title)
    tt.GetTextProperty().SetFontSize(22)
    tt.GetTextProperty().SetColor(1, 1, 1)
    tt.GetTextProperty().BoldOn()
    tt.GetPositionCoordinate().SetCoordinateSystemToNormalizedDisplay()
    tt.SetPosition(0.28, 0.02)
    ren.AddViewProp(tt)

    ren.ResetCamera()
    cam = ren.GetActiveCamera()
    cam.Elevation(elev); cam.Azimuth(azim); cam.Dolly(1.2)
    ren.ResetCameraClippingRange()

    rw = vtk.vtkRenderWindow()
    rw.SetOffScreenRendering(1)
    rw.AddRenderer(ren)
    rw.SetSize(1400, 1100)
    rw.SetMultiSamples(8)
    rw.Render()

    w2i = vtk.vtkWindowToImageFilter()
    w2i.SetInput(rw); w2i.ReadFrontBufferOff(); w2i.Update()

    wr = vtk.vtkPNGWriter()
    wr.SetFileName(out_path)
    wr.SetInputConnection(w2i.GetOutputPort())
    wr.Write()
    print(f"[Render] Saved: {out_path}")
