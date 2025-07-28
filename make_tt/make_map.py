import reco
reco = reco.Reco()

csw_info = {
  "PA" : [0,1,2,3],
  "PS" : [0,1,2,3,5,6,7],
  "ALL" : [0,1,2,3,5,6,7,9,10,22,23],
  "Special" : [0,1,2,3,5,6,7,9,10]
}


detectorpath = "calib_nuradio.json"
#detectorpath = None
station_id = 11
outfile = f"{station_id}_map_dc_avg_0610.pkl"


mappath = reco.build_travel_time_maps(detectorpath, station_id, [0], (-7000, 2000), 7000, outpath = outfile)



