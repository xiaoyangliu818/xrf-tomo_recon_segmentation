# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 15:45:10 2024

@author: lxiaoyang
"""
# this script is used to do xrf recon on 2ide tomo data
#%%
import glob
import os
import h5py
import numpy as np
import pandas as pd
from skimage import io
from matplotlib import pylab
from mpl_interactions import ipyplot as iplt
import tomopy
import math
import json
from scipy.ndimage import shift
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import random_walker
from sklearn.cluster import KMeans
#%%
#extract information from h5 file
inpath = '//micdata//data1//2ide//2021-2//Paunesku//img.dat//'
files = glob.glob(inpath+'*.h5')
theta_list = []
err = []
name = []
numof_x = []
numof_y = []
pix_sz_x = []
pix_sz_y = []
for f in files:
    try:
        h = h5py.File(f,'r')
        t = h['MAPS/extra_pvs'][1][591].astype('float32') #pv:'2xfm:m58.VAL', theta
        x = h['MAPS/Scan/requested_cols'][:][0]
        y = h['MAPS/Scan/requested_rows'][:][0]
        n = os.path.splitext(os.path.basename(f))[0]
        xdis = h['MAPS/Scan/x_axis'][:][-1]-h['MAPS/Scan/x_axis'][:][0]
        ydis = h['MAPS/Scan/y_axis'][:][-1]-h['MAPS/Scan/y_axis'][:][0]
        xpix = round(xdis*1000/x)
        ypix = round(ydis*1000/y)
        name.append(n)
        theta_list.append(t)
        numof_x.append(x)
        numof_y.append(y)
        pix_sz_x.append(xpix)
        pix_sz_y.append(ypix)
    except:
        err.append(f)
df = pd.DataFrame(zip(name,theta_list,numof_x,numof_y,pix_sz_x,pix_sz_y),columns=['scan','angle','number of x','number of y','pixel size (x)','pixel size (y)'])
df.to_csv('C://Research//OneDrive - Argonne National Laboratory//anl//userdata//2ide//scan_summary_all.csv')  

#%% create folder
sf = 'C://Research//OneDrive - Argonne National Laboratory//anl//userdata//2ide//'
scanrange = list(np.arange(319,323+1))+list(np.arange(335,397+1)) + list(np.arange(408,423+1))
notuse = [422,388,387,386,381,378]
scanrange2 = [x for x in scanrange if x not in notuse]
sf2 = f'{sf}scan{scanrange2[0]}-{scanrange2[-1]}//'
elm = ['P','S','Ca','Fe','Zn']
_ = [os.makedirs(os.path.join(sf2, e), exist_ok=True) for e in elm]

#%% 
inpath = '//micdata//data1//2ide//2021-2//Paunesku//img.dat//'
scanid = scanrange2
#sf2 = f'{sf}scan{scanrange[0]}-{scanrange[1]}//'
xmax = 400
ymax = 46
elm = ['P','S','Ca','Fe','Zn']
ang_t = []
ang = []
for e in elm:
    os.chdir(f'{sf2}{e}//')
    for s in scanid:
        s = str(s).zfill(4)
        f = f'{inpath}2xfm_{s}.mda.h5'
        h = h5py.File(f,'r')
        n = os.path.splitext(os.path.basename(f))[0]
        t = h['MAPS/extra_pvs'][1][591].astype('float32') #angle 
        t_t = h['MAPS/extra_pvs'][1][591].astype('float32') + 90.0 #angle + 90
        x = h['MAPS/Scan/requested_cols'][:][0]
        y = h['MAPS/Scan/requested_rows'][:][0]            
        '''
        h['MAPS/channel_names'][:]
        array([b'Al', b'Si', b'P', b'S', b'Cl', b'Ar', b'K', b'Ca', b'Ti', b'Cr',
               b'Mn', b'Fe', b'Co', b'Ni', b'Cu', b'Zn',
               b'Total_Fluorescence_Yield', b'Si_Si', b'Cl_Cl',
               b'COHERENT_SCT_AMPLITUDE', b'COMPTON_AMPLITUDE', b'Num_Iter',
               b'Fit_Residual', b'Sum_Elastic_Inelastic'], dtype='|S256')
        P:2; S:3; Ca:7; Fe:11; Zn:15
        h['MAPS/XRF_roi']
        <HDF5 dataset "XRF_roi": shape (24, 25, 49), type "<f4">
        '''
        idx = np.where(h['MAPS/XRF_Analyzed/Fitted/Channel_Names'][:]==e.encode())[0][0] #fitted counts_per_sec elemental names
        img = np.float32(h['MAPS/XRF_Analyzed/Fitted/Counts_Per_Sec'][idx,:,:])
        ang_t.append(t_t)
        ang.append(t)
        if x < xmax:
            p = xmax - x
            img = np.pad(img,((0,0),(round(p/2),p-round(p/2))),'constant')
        else:
            img = img
        io.imsave(f'ang{t_t}_{n}_{e}_theta{t}.tif',img)

#%%
#pad + shift projections
import json
from scipy.ndimage import shift
#alignpath = 'C://Research//OneDrive - Argonne National Laboratory//anl//userdata//2ide//scan424-481//'
alignpath = 'C://Research//OneDrive - Argonne National Laboratory//anl//userdata//2ide//scan319-423//'
os.chdir(alignpath)
img_raw = io.imread('Fe_proj.tiff')
img_raw_pad = np.zeros([img_raw.shape[0],img_raw.shape[1]+20,img_raw.shape[2]+250])
img_n = np.zeros([img_raw.shape[0],img_raw.shape[1]+20,img_raw.shape[2]+250])
for z in range(img_raw.shape[0]):               
               img_raw_pad[z] = np.pad(img_raw[z], ((10,10),(150,100)),'constant')               
#with open('Zn_align_tomviz.json','r') as alg: #for scan424-481

with open('align_tom.json','r') as alg: #for scan319-423
    al = json.load(alg)

for i,s in enumerate(al):
    img_n[i] = shift(img_raw_pad[i],[s[1],s[0]],mode='constant',cval=0)
'''
#al_pyxas = np.loadtxt('Zn_tv_pyxas_align.txt') #for scan424-481
al_pyxas = np.loadtxt('align_pyxas.txt') #for scan319-423
for i,s in enumerate(al_pyxas):
        img_n[i] = shift(img_n[i],[s[0],s[1]],mode='constant',cval=0)
#x = np.arange(0,img_n.shape[2])
#for z in range(2,img_n.shape[0]):
 #   for y in np.arange(40,45):
  #      v = img_n[z,y,:]
   #     pylab.plot


'''
io.imsave('Fe_tom_aligned.tiff',np.float32(img_n))
#%%
#find rotation center
def update(val):
    current_slice = int(slice_slider.val)
    ax.cla()  # Clear the current plot
    ax.imshow(io.imread(files[current_slice]))
    pylab.draw()
dpath = r'C:\Research\OneDrive - Argonne National Laboratory\anl\xrf\sim_test\20231020\batch\Tomo_merk\20231228'
os.chdir(dpath)    
ang = np.loadtxt('ang_Ca_3_withfly42.txt',delimiter=',')
ang_ar = sorted(np.asarray(ang))
ang_ar_radian = [math.radians(deg) for deg in ang_ar]
img_n = io.imread('Ca_original_aligned_withfly42_crop.tif')
from matplotlib.widgets import Slider    
dpath = r'C:\Research\OneDrive - Argonne National Laboratory\anl\xrf\sim_test\20231020\batch\Tomo_merk\center_test'
os.chdir(dpath)
files_old = glob.glob('*.tiff')
for fp in files_old:
    if os.path.isfile(fp):
        os.remove(fp)
cen_range = [30,50,1]
tomopy.write_center(tomo=np.float32(img_n),theta=ang_ar_radian,dpath=dpath,cen_range=cen_range,ind=69,algorithm='gridrec')
files = glob.glob('*.tiff')
num_slice = len(files)
fig, ax = pylab.subplots()
pylab.subplots_adjust(bottom=0.25)
slice_slider_ax = pylab.axes([0.2, 0.1, 0.6, 0.03])
slice_slider = Slider(slice_slider_ax, 'Slice', 0, num_slice-1, valinit=0, valstep=1)

slice_slider.on_changed(update)
ax.imshow(io.imread(files[0]))
pylab.show()
#%%
#recon
import tomopy
import math
from mpl_interactions import ipyplot as iplt
input_f = 'C://Research//OneDrive - Argonne National Laboratory//anl//userdata//2ide//scan319-423//Fe_tom_pyxas_aligned.tiff'
prj = io.imread(input_f)
ang = np.loadtxt('ang.txt')
ang_ar = sorted(np.asarray(ang))
ang_ar_radian = [math.radians(deg) for deg in ang_ar]
#one slice in y
prj_y = prj[:,18,:]
pylab.imshow(prj_y)

#sinogram
fig = pylab.figure(figsize=(12, 6))
axs = fig.subplots(5,5)
for y in np.arange(0,prj.shape[1]):
    img = prj[:,y,:]
    io.imsave(f'C://Research//OneDrive - Argonne National Laboratory//anl//userdata//2ide//scan319-423//sino//{y}.tiff',img)
    row_idx = y // 5
    col_idx = y % 5
    axs[row_idx,col_idx].imshow(img)
    axs[row_idx,col_idx].set_title(f'{y}')
pylab.tight_layout()
pylab.show()    

# find rotation center: basic find rotation axis location. The function exploits systematic artifacts 
#in reconstructed images due to shifts in the rotation center. It uses image entropy as the error metric 
#and ‘’Nelder-Mead’’ routine (of the scipy optimization module) as the optimizer [Donath:06].
for y in np.arange(0,prj.shape[1]):
    cen = tomopy.find_center(prj, ang_ar_radian,ind=y,init=100,tol=0.5)
    print(f'rot_cen_atY{y}: {cen}')
def update(val):
    current_slice = int(slice_slider.val)
    ax.cla()  # Clear the current plot
    ax.imshow(io.imread(files[current_slice]))
    pylab.draw()
from matplotlib.widgets import Slider    
dpath = 'C://Research//OneDrive - Argonne National Laboratory//anl//userdata//2ide//scan424-481//Fe_process//'
os.chdir(dpath)
files_old = glob.glob(dpath+'*.tiff')
for fp in files_old:
    if os.path.isfile(fp):
        os.remove(fp)
cen_range = [120,140,1]
tomopy.write_center(prj,ang_ar_radian,dpath,cen_range,ind=11,algorithm='gridrec')
files = glob.glob(dpath+'*.tiff')
num_slice = len(files)
fig, ax = pylab.subplots()
pylab.subplots_adjust(bottom=0.25)
slice_slider_ax = pylab.axes([0.2, 0.1, 0.6, 0.03])
slice_slider = Slider(slice_slider_ax, 'Slice', 0, num_slice-1, valinit=0, valstep=1)

slice_slider.on_changed(update)
ax.imshow(io.imread(files[0]))
pylab.show()


ini_file = 'C://Research//OneDrive - Argonne National Laboratory//anl//userdata//2ide//scan424-481//Fe_process//recon.tif'
ini_recon = io.imread(ini_file)
recon = tomopy.recon(prj,ang_ar_radian,center=133,algorithm='tv',num_gridx=)
#%%
#recon
path = r'C:\Research\OneDrive - Argonne National Laboratory\anl\xrf\sim_test\20231020\batch\Tomo_merk\20231228'
#Note: center for scan 319-423: 342
#Center for scan424-481: 282
os.chdir(path)
ini = 'Ca_aligned_crop_bkg0_manualclear_recon.tif'
img_i = io.imread(ini)
img_n = tomopy.misc.corr.remove_neg(img_n,val=0.0)
#img_n = tomopy.misc.corr.circ_mask(img_n,axis=0,ratio=0.98)
#img_n = img_n*100
#test = tomopy.prep.stripe.remove_large_stripe(tomo=img_n, snr=2, size=8)


#norm = np.linalg.norm(img_n)
#img_n_norm = img_n/norm
ang = np.loadtxt('ang_Ca_2.txt',delimiter=',')
#img_n = io.imread('testtest.tiff')
#img_n2 = tomopy.prep.normalize.normalize_roi(img_n,roi=[143,29])
#io.imsave('2.tiff',img_n2)
ang_ar = sorted(np.asarray(ang))
ang_ar_radian = [math.radians(deg) for deg in ang_ar]
recon = tomopy.recon(tomo=np.float32(img_n),theta=ang_ar_radian,center=41,algorithm='mlem',num_iter=1000,init_recon=img_i)
io.imsave('Ca_original_aligned_recon_mlem.tif',np.float32(recon))
#%%
#mask hot spots in projections
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
sf = 'C://Research//OneDrive - Argonne National Laboratory//anl//userdata//2ide//scan319-423//'
os.chdir(sf)
imf = 'Zn_tom_pyxas_aligned.tiff'
img = io.imread(imf)
for i in range(9):
    sf2 = f'Zn_it{i}//'
    os.makedirs(os.path.join(sf, sf2), exist_ok=True)
    for z in range(img.shape[0]): 
        image_max = ndi.maximum_filter(img[z], size=20, mode='constant')
    #pylab.imshow(img[47])
        coordinates = peak_local_max(img[z], min_distance=20)
        x = []
        y = []
        for c in coordinates:
            if c[0] < 45 and c[0]>38 and c[1] > 45 and c[1]<415:
                x.append(c[1])
                y.append(c[0]) 
                mean = np.mean([img[z,c[0]-5,c[1]],img[z,c[0]+5,c[1]],img[z,c[0],c[1]-5],img[z,c[0],c[1]+5]])
                img[z,c[0],c[1]] = mean
        fig = pylab.figure(figsize=(12, 6))
        pylab.imshow(img[z])
        pylab.plot(x,y, 'r.')
        pylab.savefig(f'{sf}{sf2}it{i}_{z}.tiff')
        pylab.close()
    io.imsave(f'Zn_tom_pyxas_aligned_rp_it{i}.tiff',np.float32(img))
#%%
#kmean seg
from skimage.segmentation import random_walker
from sklearn.cluster import KMeans
sf = 'C://Research//OneDrive - Argonne National Laboratory//anl//HEOs_anode//Experiment//20230623_FXI//process//HEOs_dis0p01V//'
os.chdir(sf)
f = 'pristine_z400-800_w892_h518_Mn_cropw500_h220_x120y130_bandpass.tif'
img = io.imread(f)

#k-mean method
z,y,x = img.shape
histogram, bin_edges = np.histogram(img, bins=256)
img_1d = img.reshape(z*y*x,1)     
kmeans = KMeans(n_clusters=2, init='k-means++',n_init='auto',max_iter=500,tol=1e-4, random_state=0, algorithm='lloyd').fit(img_1d)
center = kmeans.cluster_centers_
fig = pylab.figure(figsize=(12, 6))
axes = fig.subplots()
axes.plot(bin_edges[0:-1], histogram,linewidth=3)
axes.axvline(center[0,0],color='black',linewidth=2,label=f'{center[0,0]}')
axes.axvline(center[1,0],color='red',linewidth=2,label=f'{center[1,0]}')
axes.legend()

his_fit = kmeans.fit_predict(img_1d)
his_fit_2d = his_fit.reshape(z,y,x)
io.imsave(f'test.tiff',np.float32(his_fit_2d))
#%%
#watershet seg
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
import pandas as pd
sf = 'D://karen_311667//Processed//TXM//dis0.8V//'
#sf = 'C://Research//OneDrive - Argonne National Laboratory//anl//HEOs_anode//Experiment//20230623_FXI//process//HEOs_dis0p01V//'
os.chdir(sf)
f = 'dis0p01V_z400-800_y380-1100_x250-1070_Mn_cropz3-401_w680h300_x120y70_bandpass_Yenseg1p04.tif'
img = io.imread(f)
distance = ndi.distance_transform_edt(img)
mask = img>0
bin_size = 40 #pixel size
step = 1
rl = np.array(range(1, int(distance.max())+3, step))
R_dil_l = []


for rs in rl:
    # Select all the pixels with its value larger than r in distance
    center = distance>=rs
    # Dilate with a sphere kernel
    x, y, z = np.indices((2*rs+1, 2*rs+1, 2*rs+1))
    kernel = (x-rs)**2+(y-rs)**2+(z-rs)**2<rs**2
    dilation = signal.fftconvolve(center, kernel, mode='same')
    dilation[mask==False] = 0
    dilation = dilation>0.5
    R_dil = dilation.sum()-center.sum()
    R_dil_l.append(R_dil)

R_dil_l = np.array(R_dil_l)
q = R_dil_l[:-1]-R_dil_l[1:]
q_positive = q[q>=0]
q_positive = q_positive/np.sum(q_positive)
q_positive = q_positive/bin_size

r_positive = rl[-int(len(q_positive)):]*bin_size
cumulative_vol_fra = np.cumsum(q_positive * bin_size)
fig, (ax0,ax1) = plt.subplots(ncols=2, figsize=(8, 4))
ax0.plot(r_positive, q_positive,'.',linestyle='-')
ax0.set_xlabel('Feature size / nm', fontsize=22, weight='bold')
ax0.set_ylabel('Volume Fraction / bin size', fontsize=22, weight='bold')
ax1.plot(r_positive,cumulative_vol_fra,'.',linestyle='-')
ax1.set_xlabel('Feature size / nm', fontsize=22, weight='bold')
ax1.set_ylabel('Cumulative volume fraction', fontsize=22, weight='bold')

df = pd.DataFrame(zip(r_positive,q_positive,cumulative_vol_fra),columns=['feature size','volume fraction/bin size','cumulative volume fraction'])
df.to_csv(f'Feature size distribution_0p8V_Mn_20230727.csv')                                                                        
