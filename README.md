# hough-nn
Parallel Hough Transform &amp; Nearest Neighbours image algorithms
## Steps
- Save desired images to process under `images/original`: they should be ideally simple images (such as vectorized ones) where objects are clearly delimited, and shading doesn't have much impact.

  - Additionally, the background must be white - the program will consider white pixels as not being part of any objects
- Run image preprocessing: `preprocess_images.py`
  - Will preprocess all images under the aforementioned folder
- Run `main.py` - it should execute hough transform and nearest neighbors algorithms on all images within `images/original`, and plot performance metrics


