# hough-nn
Parallel Hough Transform &amp; Nearest Neighbours image algorithms
## Steps
- Save desired images to process under `images/original`: they should be ideally simple images (such as vectorized ones) where objects are clearly delimited, and shading doesn't have much impact.

  - Additionally, the background must be white - the program will consider white pixels as not being part of any objects
- Run image preprocessing: `preprocess_images.py`
  - Will preprocess all images under the aforementioned folder
- To run hough transform: `mpiexec -n 2 python hough.py`
  - Store results as a csv file in `results/hough`
- plot results using `plot_csv.py`


