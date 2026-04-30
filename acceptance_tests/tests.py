# Acceptance tests
# Running the tests from IPython
# %run -i ~/.ipython/profile_collection/acceptance_tests/tests.py

def test_fly_scan2():
    """
    Fly scan test 2.
    If db.table() and export scan complete without errors than it was successful.
    """
    print("Starting fly scan test 2")
    uid, = RE(fly_scan(relative_rot_angle=20, out_x=10, out_y=20, out_z=30, simu=True))
    print("Fly scan complete")
    print("Reading scan from databroker")
    db[uid].table(fill=True)
    print("Exporting scan")
    export_scan(db[uid].start['scan_id'])
    print("Test is complete")


def test_xanes_scan2():
    """
    Xanes scan test.
    If db.table() and export scan complete without errors than it was successful.
    """
    print("Starting xanes scan test")
    uid, = RE(xanes_scan2([8.35, 8.36, 8.37], simu=True))
    print("Fly scan complete")
    print("Reading scan from databroker ...")
    db[uid].table(fill=True)
    print("Exporting scan ...")
    export_scan(db[uid].start['scan_id'])
    print("Test is complete")

def test_fly_scan3():
    """
    Fly scan test 3.
    """
    RE(fly_scan(exposure_time=0.05, start_angle=0, relative_rot_angle=180, period=0.05, out_x=None, out_y=-10, out_z=None, out_r=0, rs=6, relative_move_flag=True, rot_first_flag=1, filters=[], add_bkg_filt_only=False, rot_back_velo=30, binning=None, move_to_ini_pos=True, simu=True, take_bkg_img=True, take_dark_img=True, close_shutter_finish=True, note="None"))


# ===========================================================================================
#                              test_fly_scan2
test_fly_scan2()

# ===========================================================================================
#                              test_xanes_scan2
test_xanes_scan2()
