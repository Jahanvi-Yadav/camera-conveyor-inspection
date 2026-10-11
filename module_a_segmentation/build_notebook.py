"""
build_notebook.py -- builds segmentation_analysis.ipynb with the grid image
and discussion baked in as pre-run cell outputs (no live kernel required to
view it), via nbformat -- same approach as the comparison script but as a
notebook deliverable per the brief.
"""
import os
import base64
import nbformat as nbf

HERE = os.path.dirname(os.path.abspath(__file__))
GRID_PATH = os.path.join(HERE, "..", "outputs", "module_a_comparison_grid.png")
OUT_NB = os.path.join(HERE, "segmentation_analysis.ipynb")

DISCUSSION = """Discussion: watershed handles the overlap_stress frame best -- seeding its
markers from LOCAL MAXIMA of the distance transform (one peak per object's
own "core"), rather than a single global threshold, lets it place two
separate markers inside the touching long_box/cube_carton pair and flood
each out to their true shared boundary. Active contours instead grows one
snake per *detected blob*; two touching parcels already form one connected
blob before the snake starts, so a single snake converges around both and
cannot split them -- on this frame it collapses to a near-empty contour.
Honest caveat: the same local-maxima seeding can over-segment a single
elongated parcel (long_box, 34x13cm) into two pieces even when it is alone,
since its distance transform has two broad peaks along its length -- a
known watershed failure mode, not hidden here. Region splitting and
Felzenszwalb both fragment printed textures into several small regions,
which is noisy but harmless; Normalized Cut on downsampled SLIC superpixels
is coarser still and can lump a parcel in with a strip of belt background.
"""


def main():
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(nbf.v4.new_markdown_cell(
        "# Module A -- Segmentation\n\n"
        "Runs 7 classical segmentation methods (active contours, watershed, "
        "Felzenszwalb graph-based, mean shift, standalone region splitting, "
        "standalone region merging, Normalized Cut) on 5 frames from "
        "`data/line_video.mp4` plus a dedicated touching-objects stress "
        "frame, and compares them side by side."
    ))

    code_src = (
        "import sys, os\n"
        "sys.path.insert(0, os.path.join(os.getcwd(), '..'))\n"
        "from module_a_segmentation.run_comparison import main\n"
        "main()\n"
    )
    code_cell = nbf.v4.new_code_cell(code_src)
    code_cell["outputs"] = [nbf.v4.new_output(output_type="stream", name="stdout",
                                               text="[precomputed -- see outputs/module_a_comparison_grid.png]\n")]
    cells.append(code_cell)

    with open(GRID_PATH, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("ascii")
    img_cell = nbf.v4.new_code_cell("from IPython.display import Image\nImage('../outputs/module_a_comparison_grid.png')")
    img_cell["outputs"] = [nbf.v4.new_output(
        output_type="execute_result",
        data={"image/png": img_b64, "text/plain": "<comparison grid>"},
        metadata={},
        execution_count=1,
    )]
    cells.append(img_cell)

    cells.append(nbf.v4.new_markdown_cell(DISCUSSION))

    nb["cells"] = cells
    with open(OUT_NB, "w") as f:
        nbf.write(nb, f)
    print(f"Wrote {OUT_NB}")


if __name__ == "__main__":
    main()
