select
sa.objid, sa.ra, sa.dec, sa.type,
sa.cmodelmag_r AS cmod_r,
sa.psfmag_u AS psf_u,
sa.psfmag_g AS psf_g,
sa.psfmag_r AS psf_r,
sa.psfmag_i AS psf_i,
sa.psfmag_z AS psf_z,
w.w1mpro AS w1,
w.w2mpro AS w2,
w.w3mpro AS w3,
w.w4mpro AS w4,
xm.match_dist AS match_dist,
pz.z, pz.zerr, pz.photoErrorClass into mydb.MyTable from PhotoPrimary as sa
JOIN wise_xmatch AS xm ON sa.objid = xm.sdss_objid
JOIN wise_allsky AS w ON xm.wise_cntr = w.cntr
LEFT JOIN photoz as pz ON sa.objid = pz.objid
WHERE
  sa.clean=1 and
  sa.specObjID=0 and
  (sa.psfmag_u < 35) and
  (sa.psfmag_g < 35) and
  (sa.psfmag_r < 35) and
  (sa.psfmag_i < 35) and
  (sa.psfmag_z < 35) and
  (sa.psfmag_u > 0) and
  (sa.psfmag_g > 0) and
  (sa.psfmag_r > 0) and
  (sa.psfmag_i > 0) and
  (sa.psfmag_z > 0) and
  (sa.cmodelmag_r <35) and
  (sa.cmodelmag_r >0) and
  (w.w1mpro<30) and
  (w.w2mpro<30) and
  (w.w3mpro<30) and
  (w.w4mpro<30) and
  (w.w1mpro>0) and
  (w.w2mpro>0) and
  (w.w3mpro>0) and
  (w.w4mpro>0)
order by
sa.objid
