print(f"Loading {__file__}...")

import tqdm
from skimage import io
# new_user()
# show_global_para()
# run_pdf()
# read_calib_file_new()
# check_latest_scan_id(init_guess=60000, search_size=100)
###################################

global img_handler

def extract_1st_proj(scan_list=[], fn_save='', clim=[0, 0.5], rot_angle=0):
    global img_handler
    img_handler = []
    if fn_save == '' and len(scan_list) >=2:
        fn_save = f'proj_1st_{scan_list[0]}_{scan_list[-1]}_angle_{rot_angle}.tiff'
    n = len(scan_list)
    img = []
    scan_failed = []
    h1 = db[int(scan_list[0])]
    tmp = list(h1.data("Andor_image", stream_name="primary"))[0]
    n_angle = len(tmp)

    ang_idx = int(rot_angle / 180. * n_angle)
    for i, sid in tqdm.tqdm(enumerate(scan_list), total=n):
        try:
            h = db[int(sid)]
            tmp = list(h.data("Andor_image", stream_name="primary"))[0]
            img_tomo = np.array(tmp[ang_idx])
            if i == 0:
                img = np.zeros((n, *img_tomo.shape))
            try:
                img_dark = np.array(list(h.data("Andor_image", stream_name="dark")))[0][0:4]
                img_dark = np.median(img_dark, axis=0)
            except:
                img_dark = np.array(list(h.data("Andor_image", stream_name="dark")))[0][0]
            try:
                img_bkg = np.array(list(h.data("Andor_image", stream_name="flat")))[0][0:4]            
                img_bkg = np.median(img_bkg, axis=0)
            except:
                img_bkg = np.array(list(h.data("Andor_image", stream_name="flat")))[0][0]
            img_norm = (img_tomo-img_dark)/(img_bkg-img_dark)
            img[i] = img_norm
        except:
            scan_failed.append(sid)
            print(f'fail to extract scan {sid}')
    img = np.array(img)
    
    if len(fn_save):
        print(f'save to {fn_save}')
        io.imsave(fn_save, img)
    #tracker = image_scrubber(img, clim)
    img_handler.append(plot3D(img))
    return img_handler, img

def load_xanes_ref(*arg):
    """
    load reference spectrum, use it as:    ref = load_xanes_ref(Ni, Ni2, Ni3)
    each spectrum is two-column array, containing: energy(1st column) and absortion(2nd column)

    It returns a dictionary, which can be used as: spectrum_ref['ref0'], spectrum_ref['ref1'] ....
    """

    num_ref = len(arg)
    assert num_ref > 1, "num of reference should larger than 1"
    spectrum_ref = {}
    for i in range(num_ref):
        spectrum_ref[f"ref{i}"] = arg[i]
    return spectrum_ref


def fit_2D_xanes_non_iter(img_xanes, eng, spectrum_ref, error_thresh=0.1):
    """
    Solve equation of Ax=b, where:

    Inputs:
    ----------
    A: reference spectrum (2-colume array: xray_energy vs. absorption_spectrum)
    X: fitted coefficient of each ref spectrum
    b: experimental 2D XANES data

    Outputs:
    ----------
    fit_coef: the 'x' in the equation 'Ax=b': fitted coefficient of each ref spectrum
    cost: cost between fitted spectrum and raw data
    """

    num_ref = len(spectrum_ref)
    spec_interp = {}
    comp = {}
    A = []
    s = img_xanes.shape
    for i in range(num_ref):
        tmp = interp1d(
            spectrum_ref[f"ref{i}"][:, 0], spectrum_ref[f"ref{i}"][:, 1], kind="cubic"
        )
        A.append(tmp(eng).reshape(1, len(eng)))
        spec_interp[f"ref{i}"] = tmp(eng).reshape(1, len(eng))
        comp[f"A{i}"] = spec_interp[f"ref{i}"].reshape(len(eng), 1)
        comp[f"A{i}_t"] = comp[f"A{i}"].T
    # e.g., spectrum_ref contains: ref1, ref2, ref3
    # e.g., comp contains: A1, A2, A3, A1_t, A2_t, A3_t
    #       A1 = ref1.reshape(110, 1)
    #       A1_t = A1.T
    A = np.squeeze(A).T
    M = np.zeros([num_ref + 1, num_ref + 1])
    for i in range(num_ref):
        for j in range(num_ref):
            M[i, j] = np.dot(comp[f"A{i}_t"], comp[f"A{j}"])
        M[i, num_ref] = 1
    M[num_ref] = np.ones((1, num_ref + 1))
    M[num_ref, -1] = 0
    # e.g.
    # M = np.array([[float(np.dot(A1_t, A1)), float(np.dot(A1_t, A2)), float(np.dot(A1_t, A3)), 1.],
    #                [float(np.dot(A2_t, A1)), float(np.dot(A2_t, A2)), float(np.dot(A2_t, A3)), 1.],
    #                [float(np.dot(A3_t, A1)), float(np.dot(A3_t, A2)), float(np.dot(A3_t, A3)), 1.],
    #                [1., 1., 1., 0.]])
    M_inv = np.linalg.inv(M)

    b_tot = img_xanes.reshape(s[0], -1)
    B = np.ones([num_ref + 1, b_tot.shape[1]])
    for i in range(num_ref):
        B[i] = np.dot(comp[f"A{i}_t"], b_tot)
    x = np.dot(M_inv, B)
    x = x[:-1]
    x[x < 0] = 0
    x_sum = np.sum(x, axis=0, keepdims=True)
    x = x / x_sum

    cost = np.sum((np.dot(A, x) - b_tot) ** 2, axis=0) / s[0]
    cost = cost.reshape(s[1], s[2])

    x = x.reshape(num_ref, s[1], s[2])
    # cost = compute_xanes_fit_cost(img_xanes, x, spec_interp)

    mask = compute_xanes_fit_mask(cost, error_thresh)
    mask = mask.reshape(s[1], s[2])
    mask_tile = np.tile(mask, (x.shape[0], 1, 1))

    x = x * mask_tile
    cost = cost * mask
    return x, cost


def fit_2D_xanes_iter(
    img_xanes,
    eng,
    spectrum_ref,
    coef0=None,
    learning_rate=0.005,
    n_iter=10,
    bounds=[0, 1],
    error_thresh=0.1,
):
    """
    Solve the equation A*x = b iteratively


    Inputs:
    -------
    img_xanes: 3D xanes image stack

    eng: energy list of xanes

    spectrum_ref: dictionary, obtained from, e.g. spectrum_ref = load_xanes_ref(Ni2, Ni3)

    coef0: initial guess of the fitted coefficient,
           it has dimention of [num_of_referece, img_xanes.shape[1], img_xanes.shape[2]]

    learning_rate: float

    n_iter: int

    bounds: [low_limit, high_limit]
          can be 'None', which give no boundary limit

    error_thresh: float
          used to generate a mask, mask[fitting_cost > error_thresh] = 0

    Outputs:
    ---------
    w: fitted 2D_xanes coefficient
       it has dimention of [num_of_referece, img_xanes.shape[1], img_xanes.shape[2]]

    cost: 2D fitting cost
    """

    num_ref = len(spectrum_ref)
    A = []
    for i in range(num_ref):
        tmp = interp1d(
            spectrum_ref[f"ref{i}"][:, 0], spectrum_ref[f"ref{i}"][:, 1], kind="cubic"
        )
        A.append(tmp(eng).reshape(1, len(eng)))
    A = np.squeeze(A).T
    Y = img_xanes.reshape(img_xanes.shape[0], -1)
    if not coef0 is None:
        W = coef0.reshape(coef0.shape[0], -1)
    w, cost = lsq_fit_iter2(A, Y, W, learning_rate, n_iter, bounds, print_flag=1)
    w = w.reshape(len(w), img_xanes.shape[1], img_xanes.shape[2])
    cost = cost.reshape(cost.shape[0], img_xanes.shape[1], img_xanes.shape[2])
    mask = compute_xanes_fit_mask(cost[-1], error_thresh)
    mask_tile = np.tile(mask, (w.shape[0], 1, 1))
    w = w * mask_tile
    mask_tile2 = np.tile(mask, (cost.shape[0], 1, 1))
    cost = cost * mask_tile2
    return w, cost


def compute_xanes_fit_cost(img_xanes, fit_coef, spec_interp):
    # compute the cost
    num_ref = len(spec_interp)
    y_fit = np.zeros(img_xanes.shape)
    for i in range(img_xanes.shape[0]):
        for j in range(num_ref):
            y_fit[i] = y_fit[i] + fit_coef[j] * np.squeeze(spec_interp[f"ref{j}"])[i]
    y_dif = np.power(y_fit - img_xanes, 2)
    cost = np.sum(y_dif, axis=0) / img_xanes.shape[0]
    return cost


def compute_xanes_fit_mask(cost, error_thresh=0.1):
    mask = np.ones(cost.shape)
    mask[cost > error_thresh] = 0
    return mask


def xanes_fit_demo():
    f = h5py.File("img_xanes_normed.h5", "r")
    img_xanes = np.array(f["img"])
    eng = np.array(f["X_eng"])
    f.close()
    img_xanes = bin_ndarray(
        img_xanes,
        (img_xanes.shape[0], int(img_xanes.shape[1] / 2), int(img_xanes.shape[2] / 2)),
    )

    Ni = np.loadtxt(
        "/nsls2/data/fxi-new/legacy/users/2018Q1/MING_Proposal_000/xanes_ref/Ni_xanes_norm.txt"
    )
    Ni2 = np.loadtxt(
        "/nsls2/data/fxi-new/legacy/users/2018Q1/MING_Proposal_000/xanes_ref/NiO_xanes_norm.txt"
    )
    Ni3 = np.loadtxt(
        "/nsls2/data/fxi-new/legacy/users/2018Q1/MING_Proposal_000/xanes_ref/LiNiO2_xanes_norm.txt"
    )

    spectrum_ref = load_xanes_ref(Ni2, Ni3)
    w1, c1 = fit_2D_xanes_non_iter(img_xanes, eng, spectrum_ref, error_thresh=0.1)
    plt.figure()
    plt.subplot(121)
    plt.imshow(w1[0])
    plt.subplot(122)
    plt.imshow(w1[1])


def temp():
    # os.mkdir('recon_image')
    scan_id = np.arange(15198, 15256)
    n = len(scan_id)
    for i in range(n):
        fn = f"fly_scan_id_{int(scan_id[i])}.h5"
        print(f"reconstructing: {fn} ... ")
        img = get_img(db[int(scan_id[i])], sli=[0, 1])
        s = img.shape
        if s[-1] > 2000:
            sli = [200, 1900]
            binning = 2
        else:
            sli = [100, 950]
            binning = 1
        rot_cen = find_rot(fn)
        recon(fn, rot_cen, sli=sli, binning=binning)
        try:
            f_recon = (
                f"recon_scan_{int(scan_id[i])}_sli_{sli[0]}_{sli[1]}_bin{binning}.h5"
            )
            f = h5py.File(f_recon, "r")
            sli_choose = int((sli[0] + sli[1]) / 2)
            img_recon = np.array(f["img"][sli_choose], dtype=np.float32)
            sid = scan_id[i]
            f.close()
            fn_img_save = f"recon_image/recon_{int(sid)}_sli_{sli_choose}.tiff"
            print(f"saving {fn_img_save}\n")
            io.imsave(fn_img_save, img_recon)
        except:
            pass


def multipos_tomo(
    exposure_time,
    x_list,
    y_list,
    z_list,
    out_x,
    out_y,
    out_z,
    out_r,
    rs,
    relative_rot_angle=185,
    period=0.05,
    relative_move_flag=0,
    traditional_sequence_flag=1,
    repeat=1,
    sleep_time=0,
    note="",
):
    n = len(x_list)
    txt = f"starting multiposition_flyscan: (repeating for {repeat} times)"
    insert_text(txt)
    for rep in range(repeat):
        for i in range(n):
            txt = f"\n################\nrepeat #{rep+1}:\nmoving to the {i+1} position: x={x_list[i]}, y={y_list[i]}, z={z_list[i]}"
            print(txt)
            insert_text(txt)
            yield from mv(zps.sx, x_list[i], zps.sy, y_list[i], zps.sz, z_list[i])
            yield from fly_scan(
                exposure_time=exposure_time,
                relative_rot_angle=relative_rot_angle,
                period=period,
                chunk_size=20,
                out_x=out_x,
                out_y=out_y,
                out_z=out_z,
                out_r=out_r,
                rs=rs,
                simu=False,
                relative_move_flag=relative_move_flag,
                traditional_sequence_flag=traditional_sequence_flag,
                note=note,
                md=None,
            )
        print(f"sleeping for {sleep_time:3.1f} s")
        yield from bps.sleep(sleep_time)


def create_lists(x0, y0, z0, dx, dy, dz, Nx, Ny, Nz):
    Ntotal = Nx * Ny * Nz
    x_list = np.zeros(Ntotal)
    y_list = np.zeros(Ntotal)
    z_list = np.zeros(Ntotal)

    j = 0
    for iz in range(Nz):
        for ix in range(Nx):
            for iy in range(Ny):
                j = iy + ix * Ny + iz * Ny * Nx  #!!!
                y_list[j] = y0 + dy * iy
                x_list[j] = x0 + dx * ix
                z_list[j] = z0 + dz * iz

    return x_list, y_list, z_list


def fan_scan(
    eng_list,
    x_list_2d,
    y_list_2d,
    z_list_2d,
    r_list_2d,
    x_list_3d,
    y_list_3d,
    z_list_3d,
    r_list_3d,
    out_x,
    out_y,
    out_z,
    out_r,
    relative_rot_angle,
    rs=3,
    exposure_time=0.05,
    chunk_size=4,
    sleep_time=0,
    repeat=1,
    relative_move_flag=True,
    note="",
):
    export_pdf(1)
    insert_text("start multiposition 2D xanes and 3D xanes")
    for i in range(repeat):
        print(f"\nrepeat # {i+1}")
        # print(f'start xanes 2D scan:')
        # yield from multipos_2D_xanes_scan2(eng_list, x_list_2d, y_list_2d, z_list_2d, r_list_2d, out_x, out_y, out_z, out_r, repeat_num=1, exposure_time=exposure_time,  sleep_time=1, chunk_size=chunk_size, simu=False, relative_move_flag=relative_move_flag, note=note, md=None)

        print("\n\nstart multi 3D xanes:")
        yield from multi_pos_xanes_3D(
            eng_list,
            x_list_3d,
            y_list_3d,
            z_list_3d,
            r_list_3d,
            exposure_time=exposure_time,
            relative_rot_angle=relative_rot_angle,
            period=exposure_time,
            out_x=out_x,
            out_y=out_y,
            out_z=out_z,
            out_r=out_r,
            rs=rs,
            simu=False,
            relative_move_flag=relative_move_flag,
            traditional_sequence_flag=1,
            note=note,
            sleep_time=0,
            repeat=1,
        )
    insert_text("finished multiposition 2D xanes and 3D xanes")
    export_pdf(1)


Ni_eng_list_101pnt = np.genfromtxt(
    "/nsls2/data/fxi-new/shared/config/xanes_ref/Ni/eng_list_Ni_xanes_standard_101pnt.txt"
)
Ni_eng_list_63pnt = np.genfromtxt(
    "/nsls2/data/fxi-new/shared/config/xanes_ref/Ni/eng_list_Ni_xanes_standard_63pnt.txt"
)
Ni_eng_list_wl = np.genfromtxt(
    "/nsls2/data/fxi-new/shared/config/xanes_ref/Ni/eng_list_Ni_xanes_standard_21pnt.txt"
)

Mn_eng_list_101pnt = np.genfromtxt(
    "/nsls2/data/fxi-new/shared/config/xanes_ref/Mn/eng_list_Mn_xanes_standard_101pnt.txt"
)
Mn_eng_list_63pnt = np.genfromtxt(
    "/nsls2/data/fxi-new/shared/config/xanes_ref/Mn/eng_list_Mn_xanes_standard_63pnt.txt"
)
Mn_eng_list_wl = np.genfromtxt(
    "/nsls2/data/fxi-new/shared/config/xanes_ref/Mn/eng_list_Mn_xanes_standard_21pnt.txt"
)

Co_eng_list_101pnt = np.genfromtxt(
    "/nsls2/data/fxi-new/shared/config/xanes_ref/Co/eng_list_Co_xanes_standard_101pnt.txt"
)
Co_eng_list_63pnt = np.genfromtxt(
    "/nsls2/data/fxi-new/shared/config/xanes_ref/Co/eng_list_Co_xanes_standard_63pnt.txt"
)
Co_eng_list_wl = np.genfromtxt(
    "/nsls2/data/fxi-new/shared/config/xanes_ref/Co/eng_list_Co_xanes_standard_21pnt.txt"
)

Fe_eng_list_101pnt = np.genfromtxt(
    "/nsls2/data/fxi-new/shared/config/xanes_ref/Fe/eng_list_Fe_xanes_standard_101pnt.txt"
)
Fe_eng_list_63pnt = np.genfromtxt(
    "/nsls2/data/fxi-new/shared/config/xanes_ref/Fe/eng_list_Fe_xanes_standard_63pnt.txt"
)
Fe_eng_list_wl = np.genfromtxt(
    "/nsls2/data/fxi-new/shared/config/xanes_ref/Fe/eng_list_Fe_xanes_standard_21pnt.txt"
)

V_eng_list_101pnt = np.genfromtxt(
    "/nsls2/data/fxi-new/shared/config/xanes_ref/V/eng_list_V_xanes_standard_101pnt.txt"
)
V_eng_list_63pnt = np.genfromtxt(
    "/nsls2/data/fxi-new/shared/config/xanes_ref/V/eng_list_V_xanes_standard_63pnt.txt"
)
V_eng_list_wl = np.genfromtxt(
    "/nsls2/data/fxi-new/shared/config/xanes_ref/V/eng_list_V_xanes_standard_21pnt.txt"
)

Cr_eng_list_101pnt = np.genfromtxt(
    "/nsls2/data/fxi-new/shared/config/xanes_ref/Cr/eng_list_Cr_xanes_standard_101pnt.txt"
)
Cr_eng_list_63pnt = np.genfromtxt(
    "/nsls2/data/fxi-new/shared/config/xanes_ref/Cr/eng_list_Cr_xanes_standard_63pnt.txt"
)
Cr_eng_list_wl = np.genfromtxt(
    "/nsls2/data/fxi-new/shared/config/xanes_ref/Cr/eng_list_Cr_xanes_standard_21pnt.txt"
)

Cu_eng_list_101pnt = np.genfromtxt(
    "/nsls2/data/fxi-new/shared/config/xanes_ref/Cu/eng_list_Cu_xanes_standard_101pnt.txt"
)
Cu_eng_list_63pnt = np.genfromtxt(
    "/nsls2/data/fxi-new/shared/config/xanes_ref/Cu/eng_list_Cu_xanes_standard_63pnt.txt"
)
Cu_eng_list_wl = np.genfromtxt(
    "/nsls2/data/fxi-new/shared/config/xanes_ref/Cu/eng_list_Cu_xanes_standard_21pnt.txt"
)

Zn_eng_list_101pnt = np.genfromtxt(
    "/nsls2/data/fxi-new/shared/config/xanes_ref/Zn/eng_list_Zn_xanes_standard_101pnt.txt"
)
Zn_eng_list_63pnt = np.genfromtxt(
    "/nsls2/data/fxi-new/shared/config/xanes_ref/Zn/eng_list_Zn_xanes_standard_63pnt.txt"
)
Zn_eng_list_wl = np.genfromtxt(
    "/nsls2/data/fxi-new/shared/config/xanes_ref/Zn/eng_list_Zn_xanes_standard_21pnt.txt"
)


# def scan_3D_2D_overnight(n):
#
#    Ni_eng_list_insitu = np.arange(8.344, 8.363, 0.001)
#    pos1 = [30, -933, -578]
#    pos2 = [-203, -1077, 563]
#    x_list = [pos1[0]]
#    y_list = [pos1[1], pos2[1]]
#    z_list = [pos1[2], pos2[2]]
#    r_list = [-71, -71]
#
#
#
#
#    #RE(multipos_2D_xanes_scan2(Ni_eng_list_insitu, x_list, y_list, z_list, [-40, -40], out_x=None, out_y=None, out_z=-2500, out_r=-90, repeat_num=1, exposure_time=0.1,  sleep_time=1, chunk_size=5, simu=False, relative_move_flag=0, note='NC_insitu'))
#
#    RE(mv(zps.sx, pos1[0], zps.sy, pos1[1], zps.sz, pos1[2], zps.pi_r, 0))
#    RE(xanes_scan2(Ni_eng_list_insitu, exposure_time=0.1, chunk_size=5, out_x=None, out_y=None, out_z=-3000, out_r=-90, simu=False, relative_move_flag=0, note='NC_insitu')
#
#    pos1 = [30, -929, -568]
#    pos_cen = [-191, -813, -563]
#    for i in range(5):
#        print(f'repeating {i+1}/{5}')
#
#        RE(mv(zps.sx, pos1[0], zps.sy, pos1[1], zps.sz, pos1[2], zps.pi_r, -72))
#        RE(xanes_3D(Ni_eng_list_insitu, exposure_time=0.1, relative_rot_angle=138, period=0.1, out_x=None, out_y=None, out_z=-3000, out_r=-90, rs=2, simu=False, relative_move_flag=0, traditional_sequence_flag=1, note='NC_insitu'))
#
#
#        RE(mv(zps.sx, pos_cen[0], zps.sy, pos_cen[1], zps.sz, pos_cen[2], zps.pi_r, 0))
#        RE(raster_2D_scan(x_range=[-1,1],y_range=[-1,1],exposure_time=0.1, out_x=None, out_y=None, out_z=-3000, out_r=-90, img_sizeX=640,img_sizeY=540,pxl=80, simu=False, relative_move_flag=0,rot_first_flag=1,note='NC_insitu'))
#
#        RE(raster_2D_xanes2(Ni_eng_list_insitu, x_range=[-1,1],y_range=[-1,1],exposure_time=0.1, out_x=None, out_y=None, out_z=-3000, out_r=-90, img_sizeX=640, img_sizeY=540, pxl=80, simu=False, relative_move_flag=0, rot_first_flag=1,note='NC_insitu'))
#
#        RE(mv(zps.sx, pos1[0], zps.sy, pos1[1], zps.sz, pos1[2], zps.pi_r, -72))
#        RE(fly_scan(exposure_time=0.1, relative_rot_angle =138, period=0.1, chunk_size=20, out_x=None, out_y=None, out_z=-3000, out_r=-90, rs=1.5, simu=False, relative_move_flag=0, traditional_sequence_flag=0, note='NC_insitu'))
#
#        RE(bps.sleep(600))
#        print('sleep for 600sec')
###############################


def mono_scan_repeatibility_test(
    pzt_cm_bender_pos_list,
    pbsl_y_pos_list,
    eng_start,
    eng_end,
    steps,
    delay_time=0.5,
    repeat=1,
):
    for ii in range(repeat):
        yield from load_cell_scan(
            pzt_cm_bender_pos_list,
            pbsl_y_pos_list,
            1,
            eng_start,
            eng_end,
            steps,
            delay_time=delay_time,
        )



def test_Andor_stage_unstage(n=500):
    global TimeStampRecord
    from bluesky import RunEngine
    from bluesky.utils import ts_msg_hook
    RE.msg_hook = ts_msg_hook
    #RE_test = RunEngine()
    
    TimeStampRecord = []
    #fsave_root = '/nsls2/data/fxi-new/legacy/users/2023Q1/commission/20230106'
    # set andor exposure time = 0.02
    # set andor acquire period = 0.05
    # set andor num image = 1000
    fsave_root = '/tmp/Andor_test'
    fsave_ts = fsave_root + '/ts_20230308.txt'
    fsave_sid = fsave_root + '/scan_id_list_20230308.txt'
    uid_list = []
    for i in range(n):
        print(f'i = {i}')
        #Andor.stage()
        #Andor.unstage()
        #uid = RE_test(count([Andor], 1))[0]
        uid = RE(count([MaranaU], 5))[0]
        sid = db[-1].start['scan_id']
        uid_list.append(sid)
        print(f'save {fsave_ts}')
        np.savetxt(fsave_ts, TimeStampRecord)
        np.savetxt(fsave_sid, uid_list, fmt='%5d')


def Read_timestampRecord(return_flag=0):
    fsave_root = '/tmp/Andor_test'
    fn_ts = fsave_root + '/ts_20230308.txt'
    fn_sid = fsave_root + '/scan_id_list_20230308.txt'
    ts = np.loadtxt(fn_ts)
    sid = np.loadtxt(fn_sid)
    ts_stage = ts[::2]
    ts_unstage = ts[1::2]
    #n_scan = len(sid)
    fig, ax = plt.subplots(1,2)
    ax[0].plot(sid, ts_stage[:], '.', label='Andor stage')
    ax[0].legend()
    ax[1].plot(sid, ts_unstage[:], '+', label='Andor unstage')
    ax[1].legend()
    if return_flag:
        return ts_stage, ts_unstage
    else:
        return 0


### a simple gui for running scan plans -XH   
from magicgui import widgets
from qtpy.QtCore import QThread, Signal
from qtpy.QtWidgets import QAbstractItemView, QTabWidget
# from PyQt5.QtCore import QObject, pyqtSignal
# from concurrent.futures import ThreadPoolExecutor
from threading import Thread

BIN_FAC = ["1x1", "2x2", "3x3", "4x4", "8x8"]


# Add this class
class UpdateHandler(QThread):
    progress_signal = Signal(str)  # Change from pyqtSignal to Signal

    def emit_progress(self, msg):
        self.progress_signal.emit(msg)


class BSPlanTab:
    def __init__(self):
        self.container = widgets.Container(widgets=[], layout='vertical')
        self._cfg_guis = {
            'scan_params_gui': None,
            'def_pos_gui': None
        }
        self._create_widgets()
        self._setup_layout()
        self.update_handler = UpdateHandler()

        @self.update_handler.progress_signal.connect
        def update_bulletin(message):
            self.bulletin.value = message

    def _create_widgets(self):
        self.plan_type = widgets.ComboBox(
            choices=[
                'tomo_zfly', 
                'tomo_grid_zfly', 
                'mosaic_zfly_scan_xh',
                'radiographic_record', 
                'dummy_scan',
                'multi_edge_xanes_zebra',

            ],
            label='Plan'
        )
        self.bulletin = widgets.Textarea(enabled=True)
        self.start_button = widgets.PushButton(text='Start Plan')
        self.stop_button = widgets.PushButton(text='Stop Plan')

        self.plan_type.changed.connect(self._plan_type_changed)
        self.start_button.changed.connect(self._start_plan)
        self.stop_button.changed.connect(self._stop_plan)
        
    def _setup_layout(self):
        self.container.extend([
            self.plan_type,
            self.bulletin,
            self.start_button,
            self.stop_button
        ])

    def _plan_type_changed(self):
        if self._cfg_guis["scan_params_gui"] is not None:
            self._cfg_guis["scan_params_gui"]._rem_member_container()        
            self._cfg_guis["scan_params_gui"]._create_widgets(
                self.plan_type.value
            )
            if self.plan_type.value == 'multi_edge_xanes_zebra':
                self._cfg_guis["def_pos_gui"].xanes_in_pos_table.enabled = True
                self._cfg_guis["def_pos_gui"].add_in_pos.enabled = True
                self._cfg_guis["def_pos_gui"].rem_in_pos.enabled = True
                self._cfg_guis["def_pos_gui"].sav_in_pos_for_later.enabled = True
                self._cfg_guis["def_pos_gui"].xanes_saved_in_pos_table.enabled = True
                self._cfg_guis["def_pos_gui"].mov_saved_in_pos.enabled = True
                self._cfg_guis["def_pos_gui"].rem_saved_in_pos.enabled = True
            else:
                self._cfg_guis["def_pos_gui"].xanes_in_pos_table.enabled = False
                self._cfg_guis["def_pos_gui"].add_in_pos.enabled = False
                self._cfg_guis["def_pos_gui"].rem_in_pos.enabled = False
                self._cfg_guis["def_pos_gui"].sav_in_pos_for_later.enabled = False
                self._cfg_guis["def_pos_gui"].xanes_saved_in_pos_table.enabled = False
                self._cfg_guis["def_pos_gui"].mov_saved_in_pos.enabled = False
                self._cfg_guis["def_pos_gui"].rem_saved_in_pos.enabled = False

    def __compose_plan_kwargs(self):
        plan_kwargs = {}
        if self.plan_type.value == 'tomo_zfly':
            plan_kwargs['scn_mode'] = int(self._cfg_guis["scan_params_gui"]._scan_mode.value)
            plan_kwargs['exp_t'] = float(self._cfg_guis["scan_params_gui"]._det_exp_t.value)
            plan_kwargs['acq_p'] = float(self._cfg_guis["scan_params_gui"]._det_acq_p.value)
            plan_kwargs['ang_s'] = float(self._cfg_guis["scan_params_gui"]._ang_s.value)
            plan_kwargs['ang_e'] = float(self._cfg_guis["scan_params_gui"]._ang_e.value)
            plan_kwargs['vel'] = float(self._cfg_guis["scan_params_gui"]._rot_vel.value)
            plan_kwargs['acc_t'] = float(self._cfg_guis["scan_params_gui"]._rot_accl.value)
            plan_kwargs['num_swing'] = int(self._cfg_guis["scan_params_gui"]._num_swings.value)
            if self._cfg_guis["def_pos_gui"].xanes_out_pos_table.choices:
                plan_kwargs['rel_out_flag'] = False
                plan_kwargs['out_pos'] = self._cfg_guis["def_pos_gui"].xanes_out_pos_table.choices[0]
            else:
                # plan_kwargs['rel_out_flag'] = True
                # plan_kwargs['out_pos'] = [1000, None, None, None]
                print("Out position is not defined")
                return None
            
            plan_kwargs['rot_back_velo'] = float(self._cfg_guis["scan_params_gui"]._rot_back_vel.value)
            plan_kwargs['bin_fac'] = BIN_FAC.index(self._cfg_guis["scan_params_gui"]._det_bin.value)
            plan_kwargs['roi'] = {
                'size_x': int(self._cfg_guis["scan_params_gui"]._det_roix_r.value),
                'size_y': int(self._cfg_guis["scan_params_gui"]._det_roiy_r.value),
                'min_x': int(self._cfg_guis["scan_params_gui"]._det_roix_s.value),
                'min_y': int(self._cfg_guis["scan_params_gui"]._det_roiy_s.value)
            }
            plan_kwargs['note'] = self._cfg_guis["scan_params_gui"]._note.value
            plan_kwargs['md'] = None
            plan_kwargs['simu'] = True
            plan_kwargs['sleep'] = float(self._cfg_guis["scan_params_gui"]._sleep_time.value)
            plan_kwargs['cam']= MaranaU
            plan_kwargs['flyer'] = tomo_flyer
            plan_kwargs['flts'] = []
            for ii, flt in enumerate(self._cfg_guis["scan_params_gui"]._flts):
                if flt.value:
                    plan_kwargs['flts'].append(ii + 1)            
        elif self.plan_type.value == 'tomo_grid_zfly':
            pass
        elif self.plan_type.value == 'mosaic_zfly_scan_xh':
            pass
        elif self.plan_type.value == 'radiographic_record':
            plan_kwargs['exp_t'] = float(self._cfg_guis["scan_params_gui"]._det_exp_t.value)
            plan_kwargs['period'] = float(self._cfg_guis["scan_params_gui"]._det_acq_p.value)
            plan_kwargs['t_span'] = float(self._cfg_guis["scan_params_gui"]._t_span.value)
            plan_kwargs['stop'] = True
            if self._cfg_guis["def_pos_gui"].xanes_out_pos_table.choices:
                plan_kwargs['relative_move_flag'] = False
                plan_kwargs['out_pos'] = self._cfg_guis["def_pos_gui"].xanes_out_pos_table.choices[0]
            else:
                print("Out position is not defined")
                return
            plan_kwargs['flts'] = []
            for ii, flt in enumerate(self._cfg_guis["scan_params_gui"]._flts):
                if flt.value:
                    plan_kwargs['flts'].append(ii + 1)
            plan_kwargs['binning'] = BIN_FAC.index(self._cfg_guis["scan_params_gui"]._det_bin.value)
            plan_kwargs['note'] = self._cfg_guis["scan_params_gui"]._note.value
            plan_kwargs['md'] = {}
            plan_kwargs['simu'] = False
            plan_kwargs['rot_first_flag'] = 1
            plan_kwargs['cam']= MaranaU
        elif self.plan_type.value == 'dummy_scan':
            plan_kwargs['exposure_time'] = float(self._cfg_guis["scan_params_gui"]._det_exp_t.value)
            plan_kwargs['ang_s'] = float(self._cfg_guis["scan_params_gui"]._ang_s.value)
            plan_kwargs['ang_e'] = float(self._cfg_guis["scan_params_gui"]._ang_e.value)
            plan_kwargs['period'] = float(self._cfg_guis["scan_params_gui"]._det_acq_p.value)
            plan_kwargs['out_pos'] = [None, None, None, None]
            plan_kwargs['rs'] = float(self._cfg_guis["scan_params_gui"]._rot_vel.value)
            plan_kwargs['flts'] = []
            for ii, flt in enumerate(self._cfg_guis["scan_params_gui"]._flts):
                if flt.value:
                    plan_kwargs['flts'].append(ii + 1)
            plan_kwargs['rot_back_velo'] = float(self._cfg_guis["scan_params_gui"]._rot_back_vel.value)
            plan_kwargs['repeat'] = int(self._cfg_guis["scan_params_gui"]._num_swings.value)
            plan_kwargs['note'] = ""
            plan_kwargs['simu'] = False
            plan_kwargs['relative_move_flag'] = 1
            plan_kwargs['cam'] = MaranaU
        elif self.plan_type.value == 'multi_edge_xanes_zebra':
            elems = self._cfg_guis["scan_params_gui"]._xanes_elem_group.value
            eng_lst = self._cfg_guis["scan_params_gui"]._xanes_eng_list.value
            flts = []
            for ii, flt in enumerate(self._cfg_guis["scan_params_gui"]._flts):
                if flt.value:
                    flts.append(ii + 1)
            
            plan_kwargs['flts'] = {}
            plan_kwargs['edge_list'] = {}
            plan_kwargs['exp_t'] = {}
            plan_kwargs['acq_p'] = {}
            for elem in elems:
                plan_kwargs['flts'][elem] = flts
                plan_kwargs['edge_list'][elem] = f"{elem}_{eng_lst.split('_')[-1]}"
                plan_kwargs['exp_t'][elem] = float(self._cfg_guis["scan_params_gui"]._det_exp_t.value)
                plan_kwargs['acq_p'][elem] = float(self._cfg_guis["scan_params_gui"]._det_acq_p.value)
            
            plan_kwargs['scan_type'] = int(self._cfg_guis["scan_params_gui"]._xanes_type.value)           
            plan_kwargs['ang_s'] = float(self._cfg_guis["scan_params_gui"]._ang_s.value)
            plan_kwargs['ang_e'] = float(self._cfg_guis["scan_params_gui"]._ang_e.value)
            plan_kwargs['vel'] = float(self._cfg_guis["scan_params_gui"]._rot_vel.value)
            plan_kwargs['acc_t'] = float(self._cfg_guis["scan_params_gui"]._rot_accl.value)
            plan_kwargs['use_gui_pos'] = True
            if self._cfg_guis["def_pos_gui"].xanes_in_pos_table.choices:
                plan_kwargs['in_pos_list'] = self._cfg_guis["def_pos_gui"].xanes_in_pos_table.choices
            else:
                plan_kwargs['in_pos_list'] = [[None, None, None, None],]

            if self._cfg_guis["def_pos_gui"].xanes_out_pos_table.choices:
                plan_kwargs['rel_out_flag'] = False
                plan_kwargs['out_pos'] = self._cfg_guis["def_pos_gui"].xanes_out_pos_table.choices[0]
            else:
                print("Out position is not defined")
                return

            plan_kwargs['rot_back_velo'] = float(self._cfg_guis["scan_params_gui"]._rot_back_vel.value)
            plan_kwargs['bin_fac'] = BIN_FAC.index(self._cfg_guis["scan_params_gui"]._det_bin.value)
            plan_kwargs['roi'] = {
                'size_x': int(self._cfg_guis["scan_params_gui"]._det_roix_r.value),
                'size_y': int(self._cfg_guis["scan_params_gui"]._det_roiy_r.value),
                'min_x': int(self._cfg_guis["scan_params_gui"]._det_roix_s.value),
                'min_y': int(self._cfg_guis["scan_params_gui"]._det_roiy_s.value)
            }
            plan_kwargs['sleep'] = float(self._cfg_guis["scan_params_gui"]._sleep_time.value)
            plan_kwargs['repeat'] = int(self._cfg_guis["scan_params_gui"]._num_swings.value)
            plan_kwargs['note'] = self._cfg_guis["scan_params_gui"]._note.value
            plan_kwargs['bulk'] = False
            plan_kwargs['md'] = None
            plan_kwargs['simu'] = False
            plan_kwargs['cam']= MaranaU
            plan_kwargs['flyer'] = tomo_flyer           
        return plan_kwargs

    def _start_plan(self):
        # print("debug 1!")
        msg = "debug 1!"
        self.update_handler.emit_progress(msg)

        def run_plan():
            print("debug 2: entering run_plan")  # Add this debug print
            plan_kwargs = self.__compose_plan_kwargs()
            if plan_kwargs is None:
                # print("Please define out position first. Quit!")
                msg = "Please define out position first. Quit!"
                self.update_handler.emit_progress(msg)
            else:
                # Setup progress monitoring
                def progress_callback(name, doc):
                    if name not in ['descriptor', 'event']:
                        print(f"{name}: {doc}")
                        self.update_handler.emit_progress(f"{name}: {doc}")    
                    # if name in ['start', 'stop']:  # Add any other types you want to see
                    #     print(f"{name}: {doc}")
                    #     self.update_handler.emit_progress(f"{name}: {doc}")      

                try:
                    token = RE.subscribe(progress_callback)

                    msg = "Scan started. Hold and don't do anything further until you see scan finish message!"
                    self.update_handler.emit_progress(msg)
                    print(f"debug 4: plan type = {self.plan_type.value}")  # Add this debug print
            
                    if self.plan_type.value == 'tomo_zfly':  
                        plan = tomo_zfly(**plan_kwargs)
                        from IPython import get_ipython
                        ipython = get_ipython()  

                        print(f"debug 5")  

                        # Store both plan and RE in user namespace
                        ipython.user_ns['_temp_plan'] = plan
                        ipython.user_ns['_temp_RE'] = RE
                        
                        print("About to execute plan...")
                        # Use the magic command to ensure execution in the main thread
                        result = ipython.magic('gui qt')  # Ensure Qt event loop is running
                        result = ipython.run_line_magic('time', '_temp_RE(_temp_plan)')
                        print(f"Run magic result: {result}")
                        
                        # Clean up
                        del ipython.user_ns['_temp_plan']
                        del ipython.user_ns['_temp_RE']
                        print("Plan execution completed")
                        print(f"debug 7")  
                    elif self.plan_type.value == 'tomo_grid_zfly':
                        RE(tomo_grid_zfly(**plan_kwargs))
                    elif self.plan_type.value == 'mosaic_zfly_scan_xh':
                        RE(mosaic_zfly_scan_xh(**plan_kwargs))
                    elif self.plan_type.value == 'radiographic_record':
                        RE(radiographic_record(**plan_kwargs))
                    elif self.plan_type.value == 'dummy_scan':
                        RE(dummy_scan(**plan_kwargs))
                    elif self.plan_type.value == 'multi_edge_xanes_zebra':
                        RE(multi_edge_xanes_zebra(**plan_kwargs))

                    msg = "Scan finished!\n"
                    self.update_handler.emit_progress(msg)                
                except Exception as e:
                    error_msg = f"Error executing plan: {str(e)}"
                    self.update_handler.emit_progress(msg)
                finally:
                    # Clean up subscription after plan completes
                    RE.unsubscribe(token)

        # Instead of creating a new thread, use the main thread
        from functools import partial
        from IPython import get_ipython
        ipython = get_ipython()
        
        # Schedule the execution in the main thread
        ipython.user_ns['_run_plan'] = run_plan
        ipython.run_cell('_run_plan()')
        print("Plan scheduled in main thread")

        # print("Starting thread...")
        # thread = Thread(target=run_plan)
        # thread.start()
        # print("Thread started")

    def _stop_plan(self):
        print("Not being implemented yet")

    def _register_cfg_guis(self, cfg_guis):
        for plan_name, gui in cfg_guis.items():
            self._cfg_guis[plan_name] = gui


class ScanParametersTab:
    def __init__(self):
        self.container = widgets.Container(
            widgets=[], 
            layout='vertical' 
        )
        self.member_container = widgets.Container(
            widgets=[], 
            layout='vertical' 
        )
        self._create_widgets()
        self._setup_layout()
        
    def _create_widgets(self, plan_name=None):
        if plan_name == 'radiographic_record':
            self.__create_radiographic_record_widgets()
        elif plan_name == 'mosaic_zfly_scan_xh':
            self.__create_mosaic_zfly_scan_xh_widgets()
        elif plan_name == 'tomo_zfly':
            self.__create_tomo_zfly_widgets()
        elif plan_name == 'tomo_grid_zfly':
            self.__create_tomo_grid_zfly_widgets()
        elif plan_name == 'dummy_scan':
            self.__create_dummy_scan_widgets()
        elif plan_name == 'multi_edge_xanes_zebra':
            self.__create_multi_edge_xanes_zebra_widgets()   
        self._setup_layout()
        
    def __create_tomo_zfly_widgets(self):
        self.__create_generic_widgets()
        self.__create_tomo_generic_widgets()

        for flt in self._flts:
            flt.value = False 
        self._det_exp_t.value = 0.006
        self._det_acq_p.value = 0.03
        self._det_bin.value = "1x1"
        self._det_roix_s.value = 1
        self._det_roix_r.value = 2048
        self._det_roiy_s.value = 1
        self._det_roiy_r.value = 2046
        self._scan_mode.value = 0
        self._ang_s.value = 0
        self._ang_e.value = 180
        self._rot_vel.value = 3
        self._rot_accl.value = 1
        self._rot_back_vel.value = 30
        self._num_swings.value = 1
        self._sleep_time.value = 0
        
        self.member_container.extend([
            self.flt_group,
            self.det_group1,
            self.det_group2,
            self._tomo_group1,
            self._tomo_group2,
            self._tomo_group3,
            self._note
        ])       

    def __create_mosaic_zfly_scan_xh_widgets(self):
        self.__create_generic_widgets()
        self.__create_tomo_generic_widgets()

        for flt in self._flts:
            flt.value = False 
        self._det_exp_t.value = 0.006
        self._det_acq_p.value = 0.03
        self._det_bin.value = "2x2"
        self._det_roix_s.value = 1
        self._det_roix_r.value = 2048
        self._det_roiy_s.value = 1
        self._det_roiy_r.value = 2046
        self._scan_mode.value = 0
        self._ang_s.value = 0
        self._ang_e.value = 180
        self._rot_vel.value = 3
        self._rot_accl.value = 1
        self._rot_back_vel.value = 30
        self._num_swings.value = 1
        self._sleep_time.value = 0
        
        self._tomo_mosaic_step_size_x = widgets.LineEdit(
            label='x step size', 
            value=0
        )
        self._tomo_mosaic_step_size_y = widgets.LineEdit(
            label='y step size', 
            value=0
        )
        self._tomo_mosaic_step_size_z = widgets.LineEdit(
            label='z step size', 
            value=0
        )
        self._tomo_mosaic_step_size_group = widgets.Container(
            layout='horizontal',
            widgets=[
                self._tomo_mosaic_step_size_x,
                self._tomo_mosaic_step_size_y,
                self._tomo_mosaic_step_size_z
            ]
        )
        
        self._tomo_mosaic_step_num_x = widgets.LineEdit(
            label='x step num', 
            value=1
        )
        self._tomo_mosaic_step_num_y = widgets.LineEdit(
            label='y step num', 
            value=1
        )
        self._tomo_mosaic_step_num_z = widgets.LineEdit(
            label='z step num', 
            value=1
        )
        self._tomo_mosaic_step_num_group = widgets.Container(
            layout='horizontal',
            widgets=[
                self._tomo_mosaic_step_num_x,
                self._tomo_mosaic_step_num_y,
                self._tomo_mosaic_step_num_z
            ]
        )
        
        self.member_container.extend([
            self.flt_group,
            self.det_group1,
            self.det_group2,
            self._tomo_group1,
            self._tomo_group2,
            self._tomo_group3,
            self._tomo_mosaic_step_size_group,
            self._tomo_mosaic_step_num_group,
            self._note
        ])     

    def __create_tomo_grid_zfly_widgets(self):
        self.__create_generic_widgets()
        self.__create_tomo_generic_widgets()

        for flt in self._flts:
            flt.value = False 
        self._det_exp_t.value = 0.006
        self._det_acq_p.value = 0.03
        self._det_bin.value = "2x2"
        self._det_roix_s.value = 1
        self._det_roix_r.value = 2048
        self._det_roiy_s.value = 1
        self._det_roiy_r.value = 2046
        self._scan_mode.value = 0
        self._ang_s.value = 0
        self._ang_e.value = 180
        self._rot_vel.value = 3
        self._rot_accl.value = 1
        self._rot_back_vel.value = 30
        self._num_swings.value = 1
        self._sleep_time.value = 0
        
        self._tomo_mosaic_step_size_x = widgets.LineEdit(
            label='x step size', 
            value=1
        )
        self._tomo_mosaic_step_size_y = widgets.LineEdit(
            label='y step size', 
            value=1
        )
        self._tomo_mosaic_step_size_z = widgets.LineEdit(
            label='z step size', 
            value=1
        )
        self._tomo_mosaic_step_size_group = widgets.Container(
            layout='horizontal',
            widgets=[
                self._tomo_mosaic_step_size_x,
                self._tomo_mosaic_step_size_y,
                self._tomo_mosaic_step_size_z
            ]
        )
        
        self._tomo_mosaic_step_num_x = widgets.LineEdit(
            label='x step num', 
            value=1
        )
        self._tomo_mosaic_step_num_y = widgets.LineEdit(
            label='y step num', 
            value=1
        )
        self._tomo_mosaic_step_num_z = widgets.LineEdit(
            label='z step num', 
            value=1
        )
        self._tomo_mosaic_step_num_group = widgets.Container(
            layout='horizontal',
            widgets=[
                self._tomo_mosaic_step_num_x,
                self._tomo_mosaic_step_num_y,
                self._tomo_mosaic_step_num_z
            ]
        )
        
        self.member_container.extend([
            self.flt_group,
            self.det_group1,
            self.det_group2,
            self._tomo_group1,
            self._tomo_group2,
            self._tomo_group3,
            self._tomo_mosaic_step_size_group,
            self._tomo_mosaic_step_num_group,
            self._note
        ]) 

    def __create_radiographic_record_widgets(self):  
        self.__create_generic_widgets()
        for flt in self._flts:
            flt.value = False 
        self._det_exp_t.value = 0.006
        self._det_acq_p.value = 1
        self._det_bin.value = "1x1"
        self._det_roix_s.value = 1
        self._det_roix_r.value = 2048
        self._det_roiy_s.value = 1
        self._det_roiy_r.value = 2046
        
        self._t_span = widgets.SpinBox(
            label='Time Span', 
            value=120, 
            tooltip='Time span in seconds'
        )

        self.member_container.extend([
            self.flt_group,
            self.det_group1,
            self.det_group2,
            self._t_span,
            self._note
        ])
    
    def __create_dummy_scan_widgets(self):
        self.__create_generic_widgets()
        self.__create_tomo_generic_widgets()

        for flt in self._flts:
            flt.value = False 
        self._det_exp_t.value = 0.006
        self._det_acq_p.value = 0.03
        self._det_bin.value = "2x2"
        self._det_roix_s.value = 1
        self._det_roix_r.value = 2048
        self._det_roiy_s.value = 1
        self._det_roiy_r.value = 2046
        self._ang_s.value = 0
        self._ang_e.value = 200
        self._rot_vel.value = 6
        self._rot_accl.value = 1
        self._rot_back_vel.value = 30
        self._num_swings.value = 20
        self._sleep_time.value = 0
        
        self.member_container.extend([
            self.flt_group,
            self.det_group1,
            self._tomo_group1,
            self._tomo_group2,
            self._tomo_group3,
            self._note
        ])

    def __create_multi_edge_xanes_zebra_widgets(self):
        self.__create_generic_widgets()
        self.__create_tomo_generic_widgets()

        for flt in self._flts:
            flt.value = False 
        self._det_exp_t.value = 0.006
        self._det_acq_p.value = 0.03
        self._det_bin.value = "2x2"
        self._det_roix_s.value = 1
        self._det_roix_r.value = 2048
        self._det_roiy_s.value = 1
        self._det_roiy_r.value = 2046
        self._scan_mode.value = 0
        self._ang_s.value = 0
        self._ang_e.value = 180
        self._rot_vel.value = 6
        self._rot_accl.value = 1
        self._rot_back_vel.value = 30
        self._num_swings.value = 1
        self._sleep_time.value = 0
        
        self._xanes_type = widgets.ComboBox(
            label="xanes type", 
            choices=["2D", "3D"], 
            value="2D"
        )
        self._xanes_eng_list = widgets.ComboBox(
            label="eng list", 
            choices=["wl_21", "wl_41", "full_63", "full_101", "..."], 
            value="wl_21" 
        )
        self._xanes_cfg_group = widgets.Container(
            layout='horizontal',
            widgets=[
                self._xanes_type,
                self._xanes_eng_list
            ])

        self._xanes_elem_group = widgets.Select(
            label="elements",
            choices=["Mn", "Fe", "Co", "Ni", "Cu", "Zn", ],
            value="Ni",
            allow_multiple=True
        )
        
        self.member_container.extend([
            self.flt_group,
            self._xanes_cfg_group,
            self._xanes_elem_group,
            self.det_group1,
            self.det_group2,
            self._tomo_group1,
            self._tomo_group2,
            self._tomo_group3,
            self._note
        ]) 

    def __create_generic_widgets(self):
        self._flts = [widgets.CheckBox(
            value=False, 
            label=f"filter {i}") for i in range(1, 5)]  
        self.flt_group = widgets.Container(
            layout='horizontal',
            widgets=self._flts
        )
        self._det_exp_t = widgets.LineEdit(
            label="exp_t (s)", 
            value=0.006,
            tooltip="exposure time in seconds"
        )
        self._det_acq_p = widgets.LineEdit(
            label="acq_p (s)", 
            value=0.03,
            tooltip="acquisition period in seconds"
        )
        self._det_bin = widgets.ComboBox(
            label="bin", 
            choices=BIN_FAC,
            value="1x1",
            tooltip="binning"
        )
        self.det_group1 = widgets.Container(
            layout='horizontal',
            widgets=[
                self._det_exp_t, 
                self._det_acq_p, 
                self._det_bin
            ]
        )
        self._det_roix_s = widgets.LineEdit(
            label="roi x start", 
            value=0,
            tooltip="roi x start in pixels"
        )
        self._det_roix_r = widgets.LineEdit(
            label="roi x range", 
            value=2048,
            tooltip="roi x range in pixels"
        )
        tmp_group1 = widgets.Container(
            layout='horizontal',
            widgets=[
                self._det_roix_s,
                self._det_roix_r
            ]
        )
        self._det_roiy_s = widgets.LineEdit(
            label="roi y start", 
            value=0,
            tooltip="roi y start in pixels"
        )
        self._det_roiy_r = widgets.LineEdit(
            label="roi y range", 
            value=2046,
            tooltip="roi y range in pixels"
        )
        tmp_group2 = widgets.Container(
            layout='horizontal',
            widgets=[
                self._det_roiy_s,
                self._det_roiy_r
            ]
        )
        self.det_group2 = widgets.Container(
            layout='vertical',
            widgets=[
                tmp_group1,
                tmp_group2
            ]
        )
        self._note = widgets.Text(label="note")

    def __create_tomo_generic_widgets(self):
        self._scan_mode = widgets.ComboBox(
            label='scan mode', 
            choices=[0, 1, 2], 
            value=0,
            tooltip='0: "standard",  a single scan in a given angle range from ang_s to ang_e; 1: "snaked: multiple files",  a back-forth rocking scan between ang_s and ang_e; each swing is saved into a file. 2: "snaked: single file",  a back-forth rocking scan between ang_s and ang_e; all scans are saved into a single file.'
        )
        self._ang_s = widgets.LineEdit(
            label='ang_s', 
            value=0, tooltip='start angle' 
        )
        self._ang_e = widgets.LineEdit(
            label='ang_e', 
            value=180, 
            tooltip='end angle' 
        )
        self._tomo_group1 = widgets.Container(
            layout='horizontal',
            widgets=[
                self._scan_mode, 
                self._ang_s, 
                self._ang_e
            ]
        )
        
        self._rot_vel = widgets.LineEdit(
            label='rot_vel', 
            value=3, 
            tooltip='rotation velocity in degrees per second' 
        )
        self._rot_accl = widgets.LineEdit(
            label='rot_accl', 
            value=1, 
            tooltip='rotation acceleration in degrees per second^2' 
        )
        self._rot_back_vel = widgets.LineEdit(
            label='rot_back_vel', 
            value=30, 
            tooltip='rotation velocity to return in degrees per second' 
        )
        self._tomo_group2 = widgets.Container(
            layout='horizontal',
            widgets=[
                self._rot_vel,
                self._rot_accl,
                self._rot_back_vel
            ]
        )

        self._num_swings = widgets.LineEdit(
            label='num_swings', 
            value=1, 
            tooltip='number of swings' 
        )
        self._sleep_time = widgets.LineEdit(
            label='sleep_time', 
            value=0, 
            tooltip='sleep time in seconds' 
        )
        self._tomo_group3 = widgets.Container(
            layout='horizontal',
            widgets=[
                self._num_swings,
                self._sleep_time
            ]
        )
       
    def _setup_layout(self):
        self.container.extend([
            self.member_container
        ])

    def _rem_member_container(self):
        self.container.clear()
        self.member_container = widgets.Container(
            widgets=[], 
            layout='vertical' 
        )


_XANES_POS_XH = {}
_XANES_IN_POS_LIST_XH = []
_XANES_OUT_POS_LIST_XH = []
_XANES_SAVED_IN_POS_XH = []


def _update_xanes_in_pos_choices(ComboBox):
    global _XANES_IN_POS_LIST_XH
    return _XANES_IN_POS_LIST_XH


def _update_xanes_out_pos_choices(ComboBox):
    global _XANES_OUT_POS_LIST_XH
    return _XANES_OUT_POS_LIST_XH

def _update_xanes_saved_in_pos_choices(ComboBox):
    global _XANES_SAVED_IN_POS_XH
    return _XANES_SAVED_IN_POS_XH


class DefPositionTab:
    def __init__(self):
        global _XANES_IN_POS_LIST_XH
        global _XANES_OUT_POS_LIST_XH
        global _XANES_POS_XH
        _XANES_IN_POS_LIST_XH = [] 
        _XANES_OUT_POS_LIST_XH = [] 
        _XANES_POS_XH = {} 

        label0 = widgets.Label(value=" define xanes scan positions ".center(100, "-"))
        self.xanes_in_pos_table = widgets.Select(
            choices=_update_xanes_in_pos_choices, name="in pos list",
        )    
        self.xanes_in_pos_table.native.setFixedWidth(300)
        self.xanes_in_pos_table.native.setFixedHeight(200)
        self.add_in_pos = widgets.PushButton(text="Add in pos")
        self.add_in_pos.native.setFixedWidth(150)
        self.rem_in_pos = widgets.PushButton(text="Rem in pos")
        self.rem_in_pos.native.setFixedWidth(150)
        self.sav_in_pos_for_later = widgets.PushButton(text="Sav in pos")
        self.sav_in_pos_for_later.native.setFixedWidth(150)
        box0 = widgets.VBox(widgets=[self.add_in_pos, self.rem_in_pos, self.sav_in_pos_for_later])
        box1 = widgets.HBox(widgets=[self.xanes_in_pos_table, box0])
        
        label1 = widgets.Label(value=" define xanes out position ".center(100, "-"))
        self.xanes_out_pos_table = widgets.Select(
            choices=_update_xanes_out_pos_choices, name="out pos"
        )
        self.xanes_out_pos_table.native.setFixedWidth(300)
        self.xanes_out_pos_table.native.setFixedHeight(50)
        self.update_out_pos = widgets.PushButton(text="Update out pos")
        self.update_out_pos.native.setFixedWidth(150)
        box2 = widgets.HBox(widgets=[self.xanes_out_pos_table, self.update_out_pos])
        self.update = widgets.PushButton(text="Update")
        self.cancel = widgets.PushButton(text="Cancel")
        box3 = widgets.HBox(widgets=[self.update, self.cancel])

        label2 = widgets.Label(value=" saved sample in positions ".center(100, "-"))
        self.xanes_saved_in_pos_table = widgets.Select(
            choices=_update_xanes_saved_in_pos_choices, name="saved in pos",
            allow_multiple=False
        )
        self.xanes_saved_in_pos_table.native.setFixedWidth(300)
        self.xanes_saved_in_pos_table.native.setFixedHeight(100)
        self.xanes_saved_in_pos_table.native.setSelectionMode(QAbstractItemView.SingleSelection)
        self.mov_saved_in_pos = widgets.PushButton(text="Move to")
        self.mov_saved_in_pos.native.setFixedWidth(150)
        self.rem_saved_in_pos = widgets.PushButton(text="Remove")
        self.rem_saved_in_pos.native.setFixedWidth(150)
        box4 = widgets.VBox(widgets=[self.mov_saved_in_pos, self.rem_saved_in_pos])
        box5 = widgets.HBox(widgets=[self.xanes_saved_in_pos_table, box4])

        label3 = widgets.Label(value=" restart everything ".center(100, "-"))
        self.restart = widgets.PushButton(text="Start Over")

        self.add_in_pos.changed.connect(self._add_in_pos)
        self.rem_in_pos.changed.connect(self._rem_in_pos)
        self.sav_in_pos_for_later.changed.connect(self._sav_in_pos_for_later)
        self.update_out_pos.changed.connect(self._update_out_pos)
        self.update.changed.connect(self._update_pos)
        self.cancel.changed.connect(self._cancel)
        self.mov_saved_in_pos.changed.connect(self._mov_saved_in_pos)
        self.rem_saved_in_pos.changed.connect(self._rem_saved_in_pos)
        self.restart.changed.connect(self._restart)
        

        self.container = widgets.VBox(
            widgets=[
                label0, 
                box1, 
                label1, 
                box2, 
                box3, 
                label2, 
                box5, 
                label3, 
                self.restart
            ] 
        )

    def _add_in_pos(self):
        global _XANES_IN_POS_LIST_XH
        _XANES_IN_POS_LIST_XH.append(
            (
                round(zps.sx.user_readback.value, 1),
                round(zps.sy.user_readback.value, 1),
                round(zps.sz.user_readback.value, 1),
                round(zps.pi_r.user_readback.value, 1),
            )
        )
        _XANES_IN_POS_LIST_XH = list(set(_XANES_IN_POS_LIST_XH))
        self.xanes_in_pos_table.reset_choices()

    def _rem_in_pos(self):
        global _XANES_IN_POS_LIST_XH
        for ii in self.xanes_in_pos_table.value:
            _XANES_IN_POS_LIST_XH.remove(ii)
        self.xanes_in_pos_table.reset_choices()

    def _sav_in_pos_for_later(self):
        global _XANES_SAVED_IN_POS_XH
        for item in _XANES_IN_POS_LIST_XH:
            _XANES_SAVED_IN_POS_XH.append(item)
        _XANES_SAVED_IN_POS_XH = list(set(_XANES_SAVED_IN_POS_XH))
        self.xanes_saved_in_pos_table.reset_choices()

    def _update_out_pos(self):
        global _XANES_OUT_POS_LIST_XH
        _XANES_OUT_POS_LIST_XH = []
        _XANES_OUT_POS_LIST_XH = [
            [
                round(zps.sx.user_readback.value, 1),
                round(zps.sy.user_readback.value, 1),
                round(zps.sz.user_readback.value, 1),
                round(zps.pi_r.user_readback.value, 1),
            ]
        ]
        self.xanes_out_pos_table.reset_choices()

    def _update_pos(self):
        global _XANES_IN_POS_LIST_XH
        global _XANES_OUT_POS_LIST_XH
        global _XANES_POS_XH
        _XANES_POS_XH["in_pos"] = _XANES_IN_POS_LIST_XH
        _XANES_POS_XH["out_pos"] = _XANES_OUT_POS_LIST_XH[0]
        print("You will use in positions:")
        for pos in _XANES_POS_XH["in_pos"]:
            print(pos)
        print(f"and out position\n{_XANES_POS_XH['out_pos']}\n\n")

    def _cancel(self):
        global _XANES_IN_POS_LIST_XH
        global _XANES_OUT_POS_LIST_XH
        global _XANES_POS_XH
        _XANES_IN_POS_LIST_XH = []
        _XANES_OUT_POS_LIST_XH = []
        _XANES_POS_XH = {}
        self.xanes_in_pos_table.reset_choices()
        self.xanes_out_pos_table.reset_choices()
        print("You cancelled all positions!\n\n")

    def _mov_saved_in_pos(self):
        confirm = widgets.Dialog(
            widgets={
                widgets.Label(value="Confirm to go to the selected position?")
            }
        ).exec()

        if confirm:
            global RE
            if self.xanes_saved_in_pos_table.value:
                RE(mv(zps.sx, self.xanes_saved_in_pos_table.value[0][0]))
                RE(mv(zps.sy, self.xanes_saved_in_pos_table.value[0][1]))
                RE(mv(zps.sz, self.xanes_saved_in_pos_table.value[0][2]))
                RE(mv(zps.pi_r, self.xanes_saved_in_pos_table.value[0][3]))

    def _rem_saved_in_pos(self):
        if self.xanes_saved_in_pos_table.value:
            global _XANES_SAVED_IN_POS_XH
            _XANES_SAVED_IN_POS_XH.remove(self.xanes_saved_in_pos_table.value[0])
            self.xanes_saved_in_pos_table.reset_choices()

    def _restart(self):
        global _XANES_IN_POS_LIST_XH
        global _XANES_OUT_POS_LIST_XH
        global _XANES_POS_XH
        global _XANES_SAVED_IN_POS_XH
        _XANES_IN_POS_LIST_XH = []
        _XANES_OUT_POS_LIST_XH = []
        _XANES_POS_XH = {}
        _XANES_SAVED_IN_POS_XH = []
        self.xanes_saved_in_pos_table.reset_choices()
        self.xanes_in_pos_table.reset_choices()
        self.xanes_out_pos_table.reset_choices()
        print("All saved positions are erased!\n\n")



class bs_gui:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(bs_gui, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        # Create main window
        self.main_window = widgets.MainWindow(layout='horizontal')
        self.main_window.native.setWindowTitle("BS GUI")
        self.main_window.native.setMinimumSize(1000, 800)
        
        # Initialize tab classes
        self.bs_plan_tab = BSPlanTab()
        self.scan_params_tab = ScanParametersTab()
        self.def_pos_tab = DefPositionTab()
        
        # Create individual containers for each tab
        self.bs_plan_container = widgets.Container(
            widgets=[self.bs_plan_tab.container],
            name='BS Plan'
        )
        self.scan_params_container = widgets.Container(
            widgets=[self.scan_params_tab.container],
            name='Scan Parameters'
        )
        self.def_pos_container = widgets.Container(
            widgets=[self.def_pos_tab.container],
            name='Define Positions'
        )

        self.bs_plan_tab._register_cfg_guis({
            'scan_params_gui': self.scan_params_tab,
            'def_pos_gui': self.def_pos_tab
        })
        
        self.tab_container = widgets.Container()
        tabs = QTabWidget()
        tabs.addTab(self.scan_params_container.native, "Scan Parameters")
        tabs.addTab(self.def_pos_container.native, "Define Positions")
        self.tab_container.native.layout().addWidget(tabs)
        
        # Add tabs to main window
        self.main_window.extend(
            [
                self.bs_plan_container,
                self.tab_container
            ]
        )
        self.main_window.show(run=False)
        self.bs_plan_tab.plan_type.value = 'dummy_scan'
        self.bs_plan_tab.plan_type.value = 'tomo_zfly'


def test_plan(**kwargs):
    x_ini = zps.sx.position
    x_out = zps.sx.position + 10
    detectors = [ic3]
    motor = [zps.sx]
    _md={"plan_name": "fly_scan"}
    _md.update(_md)
    @stage_decorator(list(detectors) + motor)
    @run_decorator(md=_md)
    def test_inner_scan():
        print("test plan starts")
        yield from mv(zps.sx, x_out)
        for ii in range(10):
            print(f"step {ii}")
            yield from bps.sleep(1)
        yield from mv(zps.sx, x_ini)
        print("test plan ends")

    yield from test_inner_scan()