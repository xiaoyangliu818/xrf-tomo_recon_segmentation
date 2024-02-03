# xrf-tomo_recon_segmentation
I used the script here to do projection information extraction, projection alignment, hot spot removal and reconstruction on a XRF tomography dataset from 2ide. The goal of the data analysis is to generate 3D information for users to do 3D printing.
1. extract the angle information from h5 file and check usable projection images
2. when aligning the projection, a hot spot was used as rotation center, however, it introduce stripes in reconstruction. I used local maximum method to remove the hot spot. I used tomviz to do projection alignment.
3. reconstruction is done by tomopy
4. segmentation: I tried kmean and watershed (used for another TXM dataset)
5. visualization: Tomviz (save as STL file)