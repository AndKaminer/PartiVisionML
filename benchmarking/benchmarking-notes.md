Want to benchmark basic background subtraction methods.

- Measure MOG2 and KNN as methods
- Measure gaussian blur and median blur as supplementary methods
- Measure choose largest and agglomerative clustering (where we then choose the largest) as methods of guaranteeing we get a single contour

Things we could potentially add:
- Sequences of these methods
- Choose all over size n and cluster those


Gaussian and clustering was way too slow bc O(n^2) complexity on contours of which there are many with gaussian
On crappy laptop, pvml took ~3 minutes.
