print(f"Loading {__file__}...")

import numpy as np
from copy import deepcopy
from scipy.ndimage.interpolation import shift
import scipy.fftpack as sf
import math
import matplotlib.pyplot as mplp
import scipy.ndimage as sn
from scipy.ndimage.filters import median_filter as medfilt
from PIL import Image
from matplotlib.widgets import Slider

def pad(img, thick, direction):

    """
    symmetrically padding the image with "0"

    Parameters:
    -----------
    img: 2d or 3d array
        2D or 3D images
    thick: int
        padding thickness for all directionsdef plot3D(data,axis=0,index_init=None):
    fig, ax = plt.subplots()
    if index_init is None:
        index_init = int(data.shape[axis]//2)
    im = ax.imshow(data.take(index_init,axis=axis))
    fig.subplots_adjust(bottom=0.15)
    axslide = fig.add_axes([0.1, 0.03, 0.8, 0.03])
    im_slider = Slider(
        ax=axslide,
        label='index',
        valmin=0,
        valmax=data.shape[axis] - 1,
        valstep=1,
        valinit=index_init,
    )
    def update(val):
        im.set_data(data.take(val,axis=axis))
        fig.canvas.draw_idle()
   
    im_slider.on_changed(update)
    plt.show()
    return im_slider
        if thick == odd, automatically increase it to thick+1
    direction: int
        0: padding in axes = 0 (2D or 3D image)
        1: padding in axes = 1 (2D or 3D image)
        2: padding in axes = 2 (3D image)

    Return:
    -------
    2d or 3d array

    """

    thick = np.int32(thick)
    if thick % 2 == 1:
        thick = thick + 1
        print("Increasing padding thickness to: {}".format(thick))

    img = np.array(img)
    s = np.array(img.shape)

    if thick == 0 or direction > 3 or s.size > 3:
        return img

    hf = np.int32(np.ceil(abs(thick) + 1) / 2)  # half size of padding thickness
    if thick > 0:
        if s.size < 3:  # 2D image
            if direction == 0:  # padding row
                pad_image = np.zeros([s[0] + thick, s[1]])
                pad_image[hf : (s[0] + hf), :] = img

            else:  # direction == 1, padding colume
                pad_image = np.zeros([s[0], s[1] + thick])
                pad_image[:, hf : (s[1] + hf)] = img

        else:  # s.size ==3, 3D image
            if direction == 0:  # padding slice
                pad_image = np.zeros([s[0] + thick, s[1], s[2]])
                pad_image[hf : (s[0] + hf), :, :] = img

            elif direction == 1:  # padding row
                pad_image = np.zeros([s[0], s[1] + thick, s[2]])
                pad_image[:, hf : (s[1] + hf), :] = img

            else:  # padding colume
                pad_image = np.zeros([s[0], s[1], s[2] + thick])
                pad_image[:, :, hf : (s[2] + hf)] = img

    else:  # thick < 0: shrink the image
        if s.size < 3:  # 2D image
            if direction == 0:  # shrink row
                pad_image = img[hf : (s[0] - hf), :]

            else:
                pad_image = img[:, hf : (s[1] - hf)]  # shrink colume

        else:  # s.size == 3, 3D image
            if direction == 0:  # shrink slice
                pad_image = img[hf : (s[0] - hf), :, :]

            elif direction == 1:  # shrink row
                pad_image = img[:, hf : (s[1] - hf), :]

            else:  # shrik colume
                pad_image = img[:, :, hf : (s[2] - hf)]
    return pad_image


def align_img(img_ref, img):
    img1_fft = np.fft.fft2(img_ref)
    img2_fft = np.fft.fft2(img)
    output = dftregistration(img1_fft, img2_fft, 100)
    row_shift = output[2]
    col_shift = output[3]
    img_shift = shift(img, [row_shift, col_shift], mode="constant", cval=0)
    return img_shift, row_shift, col_shift


"""
def align_two_img_stack(img_ref, img):
    s = img_ref.shape
    img_ali = deepcopy(img)
    for i in range(s[0]):
        img_ali[i],_, _ = align_img
"""


def align_img_stack(img, img_mask=None, select_image_index=None):
    img_align = deepcopy(img)
    n = img_align.shape[0]
    if img_mask.any() == None:
        img_mask = deepcopy(img)
    if select_image_index == None:
        for i in range(1, n):
            img_mask[i], r, c = align_img(img_mask[i - 1], img_mask[i])
            img_align[i] = shift(img_align[i], [r, c], mode="constant", cval=0)
            print("aligning #{0}, rshift:{1:3.2f}, cshift:{2:3.2f}".format(i, r, c))
    else:
        print("align image stack refereced with imgage[{}]".format(select_image_index))
        for i in range(n):
            _, r, c = align_img(img_mask[select_image_index], img_mask[i])
            img_align[i] = shift(img_align[i], [r, c], mode="constant", cval=0)
            print("aligning #{0}, rshift:{1:3.2f}, cshift:{2:3.2f}".format(i, r, c))

    return img_align


def bin_ndarray(ndarray, new_shape=None, operation="mean"):
    """
    Bins an ndarray in all axes based on the target shape, by summing or
        averaging.

    Number of output dimensions must match number of input dimensions and
        new axes must divide old ones.

    Example
    -------
    >>> m = np.arange(0,100,1).reshape((10,10))
    >>> n = bin_ndarray(m, new_shape=(5,5), operation='sum')
    >>> print(n)

    [[ 22  30  38  46  54]
     [102 110 118 126 134]
     [182 190 198 206 214]
     [262 270 278 286 294]
     [342 350 358 366 374]]

    """
    if new_shape == None:
        s = np.array(ndarray.shape)
        s1 = np.int32(s / 2)
        new_shape = tuple(s1)
    operation = operation.lower()
    if not operation in ["sum", "mean"]:
        raise ValueError("Operation not supported.")
    if ndarray.ndim != len(new_shape):
        raise ValueError("Shape mismatch: {} -> {}".format(ndarray.shape, new_shape))
    compression_pairs = [(d, c // d) for d, c in zip(new_shape, ndarray.shape)]
    flattened = [l for p in compression_pairs for l in p]
    ndarray = ndarray.reshape(flattened)
    for i in range(len(new_shape)):
        op = getattr(ndarray, operation)
        ndarray = op(-1 * (i + 1))
    return ndarray


def draw_circle(cen, r, theta=[0, 360.0]):
    th = np.linspace(theta[0] / 180.0 * np.pi, theta[1] / 180.0 * np.pi, 361)
    x = r * np.cos(th) + cen[0]
    y = r * np.sin(th) + cen[1]
    plt.plot(x, y, "r")


def get_circle_line_from_img(
    img, cen, r, pix_size=17.1, theta=[0, 360.0], f_out="circle_profile_with_fft.txt"
):
    d_th = 1 / 10.0 / r
    th = np.arange(theta[0] / 180.0 * np.pi, theta[1] / 180.0 * np.pi + d_th, d_th)
    num_data = len(th)
    x = r * np.sin(th) + cen[1]
    y = r * np.cos(th) + cen[0]

    x_int = np.int32(np.floor(x))
    x_frac = x - x_int
    y_int = np.int32(np.floor(y))
    y_frac = y - y_int

    data = []
    for i in range(num_data):
        t1 = img[x_int[i], y_int[i]] * (1 - x_frac[i]) * (1 - y_frac[i])
        t2 = img[x_int[i], y_int[i] + 1] * (1 - x_frac[i]) * y_frac[i]
        t3 = img[x_int[i] + 1, y_int[i]] * x_frac[i] * (1 - y_frac[i])
        t4 = img[x_int[i] + 1, y_int[i] + 1] * x_frac[i] * y_frac[i]
        t = t1 + t2 + t3 + t4
        data.append(t)

    line = th * r * pix_size

    plt.figure()
    plt.subplot(221)
    plt.imshow(img)
    draw_circle(cen, r, theta)

    plt.subplot(223)
    plt.plot(line, data)
    plt.title("line_profile: r={} pixels".format(r))

    data_fft = np.fft.fftshift(np.fft.fft(data))
    fs = 1 / (pix_size / 10)
    f = fs / 2 * np.linspace(-1, 1, len(data_fft))
    plt.subplot(224)
    plt.plot(f, np.abs(data_fft))
    plt.xlim([-0.04, 0.04])
    plt.ylim([-10, 300])
    plt.title("fft of line_profile")

    # combine data to sigle variable and save it
    data_comb = np.zeros([len(data), 4])
    data_comb[:, 0] = line
    data_comb[:, 1] = data
    data_comb[:, 2] = f
    data_comb[:, 3] = np.abs(data_fft)

    np.savetxt(f_out, data_comb, fmt="%3.4e")
    return data_comb


###################################################################


def dftregistration(buf1ft, buf2ft, usfac=100):
    """
           # function [output Greg] = dftregistration(buf1ft,buf2ft,usfac);
           # Efficient subpixel image registration by crosscorrelation. This code
           # gives the same precision as the FFT upsampled cross correlation in a
           # small fraction of the computation time and with reduced memory
           # requirements. It obtains an initial estimate of the
    crosscorrelation peak
           # by an FFT and then refines the shift estimation by upsampling the DFT
           # only in a small neighborhood of that estimate by means of a
           # matrix-multiply DFT. With this procedure all the image points
    are used to
           # compute the upsampled crosscorrelation.
           # Manuel Guizar - Dec 13, 2007

           # Portions of this code were taken from code written by Ann M. Kowalczyk
           # and James R. Fienup.
           # J.R. Fienup and A.M. Kowalczyk, "Phase retrieval for a complex-valued
           # object by using a low-resolution image," J. Opt. Soc. Am. A 7, 450-458
           # (1990).

           # Citation for this algorithm:
           # Manuel Guizar-Sicairos, Samuel T. Thurman, and James R. Fienup,
           # "Efficient subpixel image registration algorithms," Opt. Lett. 33,
           # 156-158 (2008).

           # Inputs
           # buf1ft    Fourier transform of reference image,
           #           DC in (1,1)   [DO NOT FFTSHIFT]
           # buf2ft    Fourier transform of image to register,
           #           DC in (1,1) [DO NOT FFTSHIFT]
           # usfac     Upsampling factor (integer). Images will be registered to
           #           within 1/usfac of a pixel. For example usfac = 20 means the
           #           images will be registered within 1/20 of a pixel.
    (default = 1)

           # Outputs
           # output =  [error,diffphase,net_row_shift,net_col_shift]
           # error     Translation invariant normalized RMS error between f and g
           # diffphase     Global phase difference between the two images (should be
           #               zero if images are non-negative).
           # net_row_shift net_col_shift   Pixel shifts between images
           # Greg      (Optional) Fourier transform of registered version of buf2ft,
           #           the global phase difference is compensated for.
    """

    # Compute error for no pixel shift
    if usfac == 0:
        CCmax = np.sum(buf1ft * np.conj(buf2ft))
        rfzero = np.sum(abs(buf1ft) ** 2)
        rgzero = np.sum(abs(buf2ft) ** 2)
        error = 1.0 - CCmax * np.conj(CCmax) / (rgzero * rfzero)
        error = np.sqrt(np.abs(error))
        diffphase = np.arctan2(np.imag(CCmax), np.real(CCmax))
        return error, diffphase

    # Whole-pixel shift - Compute crosscorrelation by an IFFT and locate the
    # peak
    elif usfac == 1:
        ndim = np.shape(buf1ft)
        m = ndim[0]
        n = ndim[1]
        CC = sf.ifft2(buf1ft * np.conj(buf2ft))
        max1, loc1 = idxmax(CC)
        rloc = loc1[0]
        cloc = loc1[1]
        CCmax = CC[rloc, cloc]
        rfzero = np.sum(np.abs(buf1ft) ** 2) / (m * n)
        rgzero = np.sum(np.abs(buf2ft) ** 2) / (m * n)
        error = 1.0 - CCmax * np.conj(CCmax) / (rgzero * rfzero)
        error = np.sqrt(np.abs(error))
        diffphase = np.arctan2(np.imag(CCmax), np.real(CCmax))
        md2 = np.fix(m / 2)
        nd2 = np.fix(n / 2)
        if rloc > md2:
            row_shift = rloc - m
        else:
            row_shift = rloc

        if cloc > nd2:
            col_shift = cloc - n
        else:
            col_shift = cloc

        ndim = np.shape(buf2ft)
        nr = int(round(ndim[0]))
        nc = int(round(ndim[1]))
        Nr = sf.ifftshift(np.arange(-np.fix(1.0 * nr / 2), np.ceil(1.0 * nr / 2)))
        Nc = sf.ifftshift(np.arange(-np.fix(1.0 * nc / 2), np.ceil(1.0 * nc / 2)))
        Nc, Nr = np.meshgrid(Nc, Nr)
        Greg = buf2ft * np.exp(
            1j * 2 * np.pi * (-1.0 * row_shift * Nr / nr - 1.0 * col_shift * Nc / nc)
        )
        Greg = Greg * np.exp(1j * diffphase)
        image_reg = sf.ifft2(Greg) * np.sqrt(nr * nc)

        # return error,diffphase,row_shift,col_shift
        return error, diffphase, row_shift, col_shift, image_reg

    # Partial-pixel shift
    else:

        # First upsample by a factor of 2 to obtain initial estimate
        # Embed Fourier data in a 2x larger array
        ndim = np.shape(buf1ft)
        m = int(round(ndim[0]))
        n = int(round(ndim[1]))
        mlarge = m * 2
        nlarge = n * 2
        CC = np.zeros([mlarge, nlarge], dtype=np.complex128)

        CC[
            int(m - np.fix(m / 2)) : int(m + 1 + np.fix((m - 1) / 2)),
            int(n - np.fix(n / 2)) : int(n + 1 + np.fix((n - 1) / 2)),
        ] = (sf.fftshift(buf1ft) * np.conj(sf.fftshift(buf2ft)))[:, :]

        # Compute crosscorrelation and locate the peak
        CC = sf.ifft2(sf.ifftshift(CC))  # Calculate cross-correlation
        max1, loc1 = idxmax(np.abs(CC))

        rloc = int(round(loc1[0]))
        cloc = int(round(loc1[1]))
        CCmax = CC[rloc, cloc]

        # Obtain shift in original pixel grid from the position of the
        # crosscorrelation peak
        ndim = np.shape(CC)
        m = ndim[0]
        n = ndim[1]

        md2 = np.fix(m / 2)
        nd2 = np.fix(n / 2)
        if rloc > md2:
            row_shift = rloc - m
        else:
            row_shift = rloc

        if cloc > nd2:
            col_shift = cloc - n
        else:
            col_shift = cloc

        row_shift = row_shift / 2
        col_shift = col_shift / 2

        # If upsampling > 2, then refine estimate with matrix multiply DFT
        if usfac > 2:
            ### DFT computation ###
            # Initial shift estimate in upsampled grid
            row_shift = 1.0 * np.round(row_shift * usfac) / usfac
            col_shift = 1.0 * np.round(col_shift * usfac) / usfac
            dftshift = np.fix(np.ceil(usfac * 1.5) / 2)
            ## Center of output array at dftshift+1
            # Matrix multiply DFT around the current shift estimate
            CC = np.conj(
                dftups(
                    buf2ft * np.conj(buf1ft),
                    np.ceil(usfac * 1.5),
                    np.ceil(usfac * 1.5),
                    usfac,
                    dftshift - row_shift * usfac,
                    dftshift - col_shift * usfac,
                )
            ) / (md2 * nd2 * usfac**2)
            # Locate maximum and map back to original pixel grid
            max1, loc1 = idxmax(np.abs(CC))
            rloc = int(round(loc1[0]))
            cloc = int(round(loc1[1]))

            CCmax = CC[rloc, cloc]
            rg00 = dftups(buf1ft * np.conj(buf1ft), 1, 1, usfac) / (
                md2 * nd2 * usfac**2
            )
            rf00 = dftups(buf2ft * np.conj(buf2ft), 1, 1, usfac) / (
                md2 * nd2 * usfac**2
            )
            rloc = rloc - dftshift
            cloc = cloc - dftshift
            row_shift = 1.0 * row_shift + 1.0 * rloc / usfac
            col_shift = 1.0 * col_shift + 1.0 * cloc / usfac

        # If upsampling = 2, no additional pixel shift refinement
        else:
            rg00 = np.sum(buf1ft * np.conj(buf1ft)) / m / n
            rf00 = np.sum(buf2ft * np.conj(buf2ft)) / m / n

        error = 1.0 - CCmax * np.conj(CCmax) / (rg00 * rf00)
        error = np.sqrt(np.abs(error))
        diffphase = np.arctan2(np.imag(CCmax), np.real(CCmax))
        # If its only one row or column the shift along that dimension has no
        # effect. We set to zero.
        if md2 == 1:
            row_shift = 0

        if nd2 == 1:
            col_shift = 0

        # Compute registered version of buf2ft
        if usfac > 0:
            ndim = np.shape(buf2ft)
            nr = ndim[0]
            nc = ndim[1]
            Nr = sf.ifftshift(np.arange(-np.fix(1.0 * nr / 2), np.ceil(1.0 * nr / 2)))
            Nc = sf.ifftshift(np.arange(-np.fix(1.0 * nc / 2), np.ceil(1.0 * nc / 2)))
            Nc, Nr = np.meshgrid(Nc, Nr)
            Greg = buf2ft * np.exp(
                1j
                * 2
                * np.pi
                * (-1.0 * row_shift * Nr / nr - 1.0 * col_shift * Nc / nc)
            )
            Greg = Greg * np.exp(1j * diffphase)
        elif (nargout > 1) & (usfac == 0):
            Greg = np.dot(buf2ft, exp(1j * diffphase))

        # mplp.figure(3)
        image_reg = sf.ifft2(Greg) * np.sqrt(nr * nc)
        # imgplot = mplp.imshow(np.abs(image_reg))

        # a_ini = np.zeros((100,100))
        # a_ini[40:59,40:59] = 1.
        # a = a_ini * np.exp(1j*15.)
        # mplp.figure(6)
        # imgplot = mplp.imshow(np.abs(a))
        # mplp.figure(3)
        # imgplot = mplp.imshow(np.abs(a)-np.abs(image_reg))
        # mplp.colorbar()

        # return error,diffphase,row_shift,col_shift,Greg
        return error, diffphase, row_shift, col_shift, image_reg


def dftups(inp, nor, noc, usfac=1, roff=0, coff=0):
    """
           # function out=dftups(in,nor,noc,usfac,roff,coff);
           # Upsampled DFT by matrix multiplies, can compute an upsampled
    DFT in just
           # a small region.
           # usfac         Upsampling factor (default usfac = 1)
           # [nor,noc]     Number of pixels in the output upsampled DFT, in
           #               units of upsampled pixels (default = size(in))
           # roff, coff    Row and column offsets, allow to shift the
    output array to
           #               a region of interest on the DFT (default = 0)
           # Recieves DC in upper left corner, image center must be in (1,1)
           # Manuel Guizar - Dec 13, 2007
           # Modified from dftus, by J.R. Fienup 7/31/06

           # This code is intended to provide the same result as if the following
           # operations were performed
           #   - Embed the array "in" in an array that is usfac times larger in each
           #     dimension. ifftshift to bring the center of the image to (1,1).
           #   - Take the FFT of the larger array
           #   - Extract an [nor, noc] region of the result. Starting with the
           #     [roff+1 coff+1] element.

           # It achieves this result by computing the DFT in the output
    array without
           # the need to zeropad. Much faster and memory efficient than the
           # zero-padded FFT approach if [nor noc] are much smaller than
    [nr*usfac nc*usfac]
    """

    ndim = np.shape(inp)
    nr = int(round(ndim[0]))
    nc = int(round(ndim[1]))
    noc = int(round(noc))
    nor = int(round(nor))

    # Compute kernels and obtain DFT by matrix products
    a = np.zeros([nc, 1])
    a[:, 0] = ((sf.ifftshift(np.arange(nc))) - np.floor(1.0 * nc / 2))[:]
    b = np.zeros([1, noc])
    b[0, :] = (np.arange(noc) - coff)[:]
    kernc = np.exp((-1j * 2 * np.pi / (nc * usfac)) * np.dot(a, b))
    nndim = kernc.shape
    # print nndim

    a = np.zeros([nor, 1])
    a[:, 0] = (np.arange(nor) - roff)[:]
    b = np.zeros([1, nr])
    b[0, :] = (sf.ifftshift(np.arange(nr)) - np.floor(1.0 * nr / 2))[:]
    kernr = np.exp((-1j * 2 * np.pi / (nr * usfac)) * np.dot(a, b))
    nndim = kernr.shape
    # print nndim

    return np.dot(np.dot(kernr, inp), kernc)


def idxmax(data):
    ndim = np.shape(data)
    # maxd = np.max(data)
    maxd = np.max(np.abs(data))
    # t1 = mplp.mlab.find(np.abs(data) == maxd)
    t1 = np.argmin(np.abs(np.abs(data) - maxd))
    idx = np.zeros(
        [
            len(ndim),
        ]
    )
    for ii in range(len(ndim) - 1):
        t1, t2 = np.modf(1.0 * t1 / np.prod(ndim[(ii + 1) :]))
        idx[ii] = t2
        t1 *= np.prod(ndim[(ii + 1) :])
    idx[np.size(ndim) - 1] = t1

    return maxd, idx


def flip_conj(tmp):
    # ndims = np.shape(tmp)
    # nx = ndims[0]
    # ny = ndims[1]
    # nz = ndims[2]
    # tmp_twin = np.zeros([nx,ny,nz]).astype(complex)
    # for i in range(0,nx):
    #   for j in range(0,ny):
    #      for k in range(0,nz):
    #         i_tmp = nx - 1 - i
    #         j_tmp = ny - 1 - j
    #         k_tmp = nz - 1 - k
    #         tmp_twin[i,j,k] = tmp[i_tmp,j_tmp,k_tmp].conj()
    # return tmp_twin

    tmp_fft = sf.ifftshift(sf.ifftn(sf.fftshift(tmp)))
    return sf.ifftshift(sf.fftn(sf.fftshift(np.conj(tmp_fft))))


def check_conj(ref, tmp, threshold_flag, threshold, subpixel_flag):
    ndims = np.shape(ref)
    nx = ndims[0]
    ny = ndims[1]
    nz = ndims[2]

    if threshold_flag == 1:
        ref_tmp = np.zeros((nx, ny, nz))
        index = np.where(np.abs(ref) >= threshold * np.max(np.abs(ref)))
        ref_tmp[index] = 1.0
        tmp_tmp = np.zeros((nx, ny, nz))
        index = np.where(np.abs(tmp) >= threshold * np.max(np.abs(tmp)))
        tmp_tmp[index] = 1.0
        tmp_conj = flip_conj(tmp_tmp)
    else:
        ref_tmp = ref
        tmp_tmp = tmp
        tmp_conj = flip_conj(tmp)

    tmp_tmp = subpixel_align(ref_tmp, tmp_tmp, threshold_flag, threshold, subpixel_flag)
    tmp_conj = subpixel_align(
        ref_tmp, tmp_conj, threshold_flag, threshold, subpixel_flag
    )

    cc_1 = sf.ifftn(ref_tmp * np.conj(tmp_tmp))
    cc1 = np.max(cc_1.real)
    # cc1 = np.max(np.abs(cc_1))
    cc_2 = sf.ifftn(ref_tmp * np.conj(tmp_conj))
    cc2 = np.max(cc_2.real)
    # cc2 = np.max(np.abs(cc_2))
    print("{0}, {1}".format(cc1, cc2))
    if cc1 > cc2:
        return 0
    else:
        return 1


def subpixel_align(ref, tmp, threshold_flag, threshold, subpixel_flag):
    ndims = np.shape(ref)
    if np.size(ndims) == 3:
        nx = ndims[0]
        ny = ndims[1]
        nz = ndims[2]

        if threshold_flag == 1:
            ref_tmp = np.zeros((nx, ny, nz))
            index = np.where(np.abs(ref) >= threshold * np.max(np.abs(ref)))
            ref_tmp[index] = 1.0
            tmp_tmp = np.zeros((nx, ny, nz))
            index = np.where(np.abs(tmp) >= threshold * np.max(np.abs(tmp)))
            tmp_tmp[index] = 1.0
            ref_fft = sf.ifftn(sf.fftshift(ref_tmp))
            tmp_fft = sf.ifftn(sf.fftshift(tmp_tmp))
            real_fft = sf.ifftn(sf.fftshift(tmp))
        else:
            ref_fft = sf.ifftn(sf.fftshift(ref))
            tmp_fft = sf.ifftn(sf.fftshift(tmp))

        nest = np.mgrid[0:nx, 0:ny, 0:nz]

        result = dftregistration(ref_fft[:, :, 0], tmp_fft[:, :, 0], usfac=100)
        e, p, cl, r, array_shift = result
        x_shift_1 = cl
        y_shift_1 = r
        result = dftregistration(
            ref_fft[:, :, nz - 1], tmp_fft[:, :, nz - 1], usfac=100
        )
        e, p, cl, r, array_shift = result
        x_shift_2 = cl
        y_shift_2 = r

        result = dftregistration(ref_fft[:, 0, :], tmp_fft[:, 0, :], usfac=100)
        e, p, cl, r, array_shift = result
        x_shift_3 = cl
        z_shift_1 = r
        result = dftregistration(
            ref_fft[:, ny - 1, :], tmp_fft[:, ny - 1, :], usfac=100
        )
        e, p, cl, r, array_shift = result
        x_shift_4 = cl
        z_shift_2 = r

        result = dftregistration(ref_fft[0, :, :], tmp_fft[0, :, :], usfac=100)
        e, p, cl, r, array_shift = result
        y_shift_3 = cl
        z_shift_3 = r
        result = dftregistration(
            ref_fft[nx - 1, :, :], tmp_fft[nx - 1, :, :], usfac=100
        )
        e, p, cl, r, array_shift = result
        y_shift_4 = cl
        z_shift_4 = r

        if subpixel_flag == 1:
            x_shift = (x_shift_1 + x_shift_2 + x_shift_3 + x_shift_4) / 4.0
            y_shift = (y_shift_1 + y_shift_2 + y_shift_3 + y_shift_4) / 4.0
            z_shift = (z_shift_1 + z_shift_2 + z_shift_3 + z_shift_4) / 4.0
        else:
            x_shift = np.floor(
                (x_shift_1 + x_shift_2 + x_shift_3 + x_shift_4) / 4.0 + 0.5
            )
            y_shift = np.floor(
                (y_shift_1 + y_shift_2 + y_shift_3 + y_shift_4) / 4.0 + 0.5
            )
            z_shift = np.floor(
                (z_shift_1 + z_shift_2 + z_shift_3 + z_shift_4) / 4.0 + 0.5
            )

        print("x, y, z shift: {0}, {1}, {2}".format(x_shift, y_shift, z_shift))

        if threshold_flag == 1:
            tmp_fft_new = sf.ifftshift(real_fft) * np.exp(
                1j
                * 2
                * np.pi
                * (
                    -1.0 * x_shift * (nest[0, :, :, :] - nx / 2.0) / (nx)
                    - y_shift * (nest[1, :, :, :] - ny / 2.0) / (ny)
                    - z_shift * (nest[2, :, :, :] - nz / 2.0) / (nz)
                )
            )
        else:
            tmp_fft_new = sf.ifftshift(tmp_fft) * np.exp(
                1j
                * 2
                * np.pi
                * (
                    -1.0 * x_shift * (nest[0, :, :, :] - nx / 2.0) / (nx)
                    - y_shift * (nest[1, :, :, :] - ny / 2.0) / (ny)
                    - z_shift * (nest[2, :, :, :] - nz / 2.0) / (nz)
                )
            )

    if np.size(ndims) == 2:
        nx = ndims[0]
        ny = ndims[1]

        if threshold_flag == 1:
            ref_tmp = np.zeros((nx, ny))
            index = np.where(np.abs(ref) >= threshold * np.max(np.abs(ref)))
            ref_tmp[index] = 1.0
            tmp_tmp = np.zeros((nx, ny))
            index = np.where(np.abs(tmp) >= threshold * np.max(np.abs(tmp)))
            tmp_tmp[index] = 1.0

            ref_fft = sf.ifftn(sf.fftshift(ref_tmp))
            mp_fft = sf.ifftn(sf.fftshift(tmp_tmp))
            real_fft = sf.ifftn(sf.fftshift(tmp))
        else:
            ref_fft = sf.ifftn(sf.fftshift(ref))
            tmp_fft = sf.ifftn(sf.fftshift(tmp))

        nest = np.mgrid[0:nx, 0:ny]

        result = dftregistration(ref_fft[:, :], tmp_fft[:, :], usfac=100)
        e, p, cl, r, array_shift = result
        x_shift = cl
        y_shift = r

        if subpixel_flag == 1:
            x_shift = x_shift
            y_shift = y_shift
        else:
            x_shift = np.floor(x_shift + 0.5)
            y_shift = np.floor(y_shift + 0.5)

        print("x, y shift: {0}, {1}".format(x_shift, y_shift))

        if threshold_flag == 1:
            tmp_fft_new = sf.ifftshift(real_fft) * np.exp(
                1j
                * 2
                * np.pi
                * (
                    -1.0 * x_shift * (nest[0, :, :] - nx / 2.0) / (nx)
                    - y_shift * (nest[1, :, :] - ny / 2.0) / (ny)
                )
            )
        else:
            tmp_fft_new = sf.ifftshift(tmp_fft) * np.exp(
                1j
                * 2
                * np.pi
                * (
                    -1.0 * x_shift * (nest[0, :, :] - nx / 2.0) / (nx)
                    - y_shift * (nest[1, :, :] - ny / 2.0) / (ny)
                )
            )

    return sf.ifftshift(sf.fftn(sf.fftshift(tmp_fft_new))), x_shift, y_shift


def remove_phase_ramp(tmp, threshold_flag, threshold, subpixel_flag):
    tmp_tmp, x_shift, y_shift = subpixel_align(
        sf.ifftshift(sf.ifftn(sf.fftshift(np.abs(tmp)))),
        sf.ifftshift(sf.ifftn(sf.fftshift(tmp))),
        threshold_flag,
        threshold,
        subpixel_flag,
    )
    tmp_new = sf.ifftshift(sf.fftn(sf.fftshift(tmp_tmp)))
    phase_tmp = np.angle(tmp_new)
    ph_offset = np.mean(phase_tmp[np.where(np.abs(tmp) >= threshold)])
    phase_tmp = np.angle(tmp_new) - ph_offset
    return np.abs(tmp) * np.exp(1j * phase_tmp)


def pixel_shift(array, x_shift, y_shift, z_shift):
    nx, ny, nz = np.shape(array)
    tmp = sf.ifftshift(sf.ifftn(sf.fftshift(array)))
    nest = np.mgrid[0:nx, 0:ny, 0:nz]
    tmp = tmp * np.exp(
        1j
        * 2
        * np.pi
        * (
            -1.0 * x_shift * (nest[0, :, :, :] - nx / 2.0) / (nx)
            - y_shift * (nest[1, :, :, :] - ny / 2.0) / (ny)
            - z_shift * (nest[2, :, :, :] - nz / 2.0) / (nz)
        )
    )
    return sf.ifftshift(sf.fftn(sf.fftshift(tmp)))


def pixel_shift_2d(array, x_shift, y_shift):
    nx, ny = np.shape(array)
    tmp = sf.ifftshift(sf.ifftn(sf.fftshift(array)))
    nest = np.mgrid[0:nx, 0:ny]
    tmp = tmp * np.exp(
        1j
        * 2
        * np.pi
        * (
            -1.0 * x_shift * (nest[0, :, :] - nx / 2.0) / (nx)
            - y_shift * (nest[1, :, :] - ny / 2.0) / (ny)
        )
    )
    return sf.ifftshift(sf.fftn(sf.fftshift(tmp)))


def rm_phase_ramp_manual_2d(array, x_shift, y_shift):
    nx, ny = np.shape(array)
    nest = np.mgrid[0:nx, 0:ny]
    tmp = array * np.exp(
        1j
        * 2
        * np.pi
        * (
            -1.0 * x_shift * (nest[0, :, :] - nx / 2.0) / (nx)
            - y_shift * (nest[1, :, :] - ny / 2.0) / (ny)
        )
    )
    return tmp



def plot3D(data,axis=0,index_init=None):
    fig, ax = plt.subplots()
    if index_init is None:
        index_init = int(data.shape[axis]//2)
    im = ax.imshow(data.take(index_init,axis=axis))
    fig.subplots_adjust(bottom=0.15)
    axslide = fig.add_axes([0.1, 0.03, 0.8, 0.03])
    im_slider = Slider(
        ax=axslide,
        label='index',
        valmin=0,
        valmax=data.shape[axis] - 1,
        valstep=1,
        valinit=index_init,
    )
    def update(val):
        im.set_data(data.take(val,axis=axis))
        fig.canvas.draw_idle()
   
    im_slider.on_changed(update)
    plt.show()
    return im_slider


if __name__ == "__main__":
    pass
