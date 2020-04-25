from netCDF4 import Dataset
import datetime
import cartopy.crs as ccrs
import cartopy.feature as cfeat
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import num2date
import metpy.calc as mpcalc
from metpy.units import units
import scipy.ndimage as ndimage
import os
import os.path

class ReanalysisData(object):
    
    def __init__(self, nc_file):
        self.datafromfile = Dataset(nc_file, 'r', format='NETCDF4')
        
    def retrieve_data(self):
        
        datafromfile = self.datafromfile
        time_var = datafromfile.variables['time']
        time_vals = num2date(time_var[:].squeeze(), time_var.units)
        lons = datafromfile['longitude'][:].squeeze()
        lats = datafromfile['latitude'][:].squeeze()
        if 'level' in datafromfile.variables.keys():
            levels = datafromfile['level'][:]
            coord_vars = ['longitude', 'latitude', 'time', 'level']
            varData = {key: datafromfile[key][:].squeeze() for key in datafromfile.variables.keys() if key not in coord_vars}
            data_dict = {'lons': lons, 'lats': lats, 'time': time_vals, 'levels': levels,
                         'data': varData}
        else:
            coord_vars = ['longitude', 'latitude', 'time']
            varData={key: datafromfile[key][:].squeeze() for key in datafromfile.variables.keys() if key not in coord_vars}
            data_dict = {'lons': lons, 'lats': lats, 'time': time_vals,
                        'data': varData}
    
        self.data_dict = data_dict
        
    def formatting_data(self):
        
        DictForConvertingVars = {'r': lambda x: x/100, 't': lambda x: x*units.K,
                                'u': lambda x: x, 'v': lambda x: x, 'z': lambda x: x/9.80665,
                                't2m': lambda x: (x-273.15)*1.8 + 32, 'd2m': lambda x: (x-273.15)*1.8+32,
                                'msl': lambda x: x/100, 'vo': lambda x: x, 'u10': lambda x: x,
                                'v10': lambda x: x}
        self.retrieve_data()
        data_dict = self.data_dict
        varData = data_dict['data']
        converted = {k: DictForConvertingVars[k](v) for k, v in varData.items()}
        if 'levels' in data_dict.keys():
            full_dict = {'data': converted, 'coords': {'time': data_dict['time'],
                                                  'lons': data_dict['lons'],
                                                  'lats': data_dict['lats'],
                                                  'levels': data_dict['levels']}}
        else:
            full_dict = {'data': converted, 'coords': {'time': data_dict['time'],
                                                      'lons': data_dict['lons'],
                                                      'lats': data_dict['lats']}}
            
        return full_dict
            
def plot_backgroundmap(plot_proj, scale, extent):
    
    fig = plt.figure(figsize=(15,15))
    ax = plt.axes(projection=plot_proj)
    ax.add_feature(cfeat.STATES.with_scale(scale), facecolor='gray', alpha=0.4)
    ax.add_feature(cfeat.BORDERS.with_scale(scale), facecolor='none', edgecolor='black')
    ax.add_feature(cfeat.LAND.with_scale(scale), facecolor='none', edgecolor='black')
    ax.set_extent(extent)
    return fig, ax

def calc_thetae_advec(dx, dy, var_dict, pres):
    
    wind_vars = ['u', 'v']
    wind_dict = {key: var_dict[pres][key] for key in wind_vars}
    tmp_var = 't'
    rh_var = 'r'
    tmpk = var_dict[pres][tmp_var]
    rh = var_dict[pres][rh_var]
    rh[rh>1.0] = 1.0
    dp = mpcalc.dewpoint_rh(tmpk, rh.data)
    theta_e = mpcalc.equivalent_potential_temperature(pres*units.hPa, tmpk, dp)
    times, _, _ = theta_e.shape
    theta_e_advec = np.zeros(theta_e.shape)
    for i in range(times):
        theta_e_advec[i] = mpcalc.advection(theta_e[i], [wind_dict['u'][i], wind_dict['v'][i]],
                                           [dx, dy])
        
    return theta_e_advec

def plot_thetae_advec(ax, lon2d, lat2d, advec, kwargs_dict):
    
    advec_cons = ax.contourf(lon2d, lat2d, advec*3600*3,
                            levels=kwargs_dict['advec_levels'], cmap=kwargs_dict['cmap'],
                            extend='both', transform=kwargs_dict['data_proj'])
    
    units = 'K/3hrs'
    return ax, advec_cons, units

def calc_vort_advec(dx, dy, lat2d, var_dict, pres):
    
    wind_vars = ['u', 'v']
    wind_dict = {key: var_dict[pres][key] for key in wind_vars}
    latradians = lat2d*np.pi/180
    coriolis = mpcalc.coriolis_parameter(latradians)
    abs_vort = var_dict[pres]['vo'] + coriolis
    times, _,_ = abs_vort.shape
    vort_advec = np.zeros(abs_vort.shape)
    for i in range(times):
        smabs_vort = ndimage.gaussian_filter(abs_vort[i], sigma=2)
        vort_advec[i] = mpcalc.advection(smabs_vort, [wind_dict['u'][i], wind_dict['v'][i]],
                                        [dx, dy])
    return vort_advec

def plot_vort_advec(ax, lon2d, lat2d, advec, kwargs_dict):
    
    advec_cons = ax.contourf(lon2d, lat2d, advec*1e9,
                            levels=kwargs_dict['advec_levels'], cmap=kwargs_dict['cmap'],
                            extend='both', transform=kwargs_dict['data_proj'])
    
    units = '10$^-9$ s$^{-1}$'
    
    return ax, advec_cons, units

def calc_qvectors():
    return

def calc_moist_advec(dx, dy, var_dict, pres):
    wind_vars = ['u', 'v']
    wind_dict = {key: var_dict[pres][key] for key in wind_vars}
    tmpk = var_dict[pres]['t']
    rh = var_dict[pres]['r']
    rh[rh>1.0]=1.0
    mix_rat = mpcalc.mixing_ratio_from_relative_humidity(rh.data, tmpk, pres*units.hPa)*1000
    times, _, _ = mix_rat.shape
    moist_advec = np.zeros(mix_rat.shape)
    for i in range(times):
        moist_advec[i] = ndimage.gaussian_filter(mpcalc.advection(mix_rat[i].m, [wind_dict['u'][i], wind_dict['v'][i]],
                                         [dx, dy]), sigma=1)
    
    return moist_advec

def plot_moist_advec(ax, lon2d, lat2d, advec, kwargs_dict):
    
    advec_cons = ax.contourf(lon2d, lat2d, advec*3600*3,
                            levels=kwargs_dict['advec_levels'], cmap=kwargs_dict['cmap'],
                            extend='both', transform=kwargs_dict['data_proj'])
    
    units = 'g/kg/3hrs'
    return ax, advec_cons, units

def plot_qvectors():
    return

def calc_div(dx, dy, var_dict, pres):
    
    wind_vars = ['u', 'v']
    wind_dict = {key: var_dict[pres][key] for key in wind_vars}
    diverg = np.zeros(wind_dict['u'].shape)
    times,_,_ = diverg.shape
    for i in range(times):
        diverg[i] = ndimage.gaussian_filter(mpcalc.divergence(wind_dict['u'][i], wind_dict['v'][i], dx, dy), sigma=2)
        
    return diverg

def plot_div():
    return

def plot_kwargs(pres, plot_type):
    
    heightsDict = {925: np.arange(0, 3000, 30), 
                  850: np.arange(0, 3000, 30),
                  700: np.arange(0, 6120, 30),
                  500: np.arange(0, 6120, 60),
                  300: np.arange(0, 12000, 120)}
    
    advec_levelsDict = {'thetae_advec': list(range(-18, 0, 2))+list(range(2, 20, 2)),
                       'vort_advec': list(range(-30, 0, 2))+list(range(2, 32, 2)),
                       'moist_advec': list(np.arange(-3, 0.05, 0.1))+list(np.arange(0.05, 3.1, 0.1))}
    
    scale = '10m'
    plot_proj = ccrs.LambertConformal(central_longitude=-100, central_latitude=40)
    data_proj = ccrs.PlateCarree()
    extent = [-125, -65, 20, 50]
    levels_height = heightsDict[pres]
    levels_isall = np.arange(2, 28, 2)
    iso_levels = np.arange(940, 1104, 4)
    fontsize=12
    wind_slice = (slice(None, None, 9), slice(None, None, 9))
    
    kwargs_dict = dict(scale=scale, plot_proj=plot_proj,
                      data_proj=data_proj, extent=extent, levels_height=heightsDict[pres],
                      levels_isall = levels_isall, 
                       isobar_levels=iso_levels, advec_levels=advec_levelsDict[plot_type],
                       fontsize=12, wind_slice=wind_slice)
    
    return kwargs_dict
    
            
def plot_composite_chart(timerange, coords_dict, sfc_dict, pres_dict, pres1, pres2, plot_type):
    
    calc_dict = {'calc_thetae_advec': calc_thetae_advec, 'calc_vort_advec': calc_vort_advec,
                 'calc_moist_advec': calc_moist_advec, 'calc_qvectors': calc_qvectors,
                'calc_div': calc_div}
    plot_dict = {'plot_thetae_advec': plot_thetae_advec, 'plot_vort_advec': plot_vort_advec, 
                 'plot_moist_advec': plot_moist_advec, 'plot_qvectors': plot_qvectors,
                'plot_div': plot_div}
    
    title_dict = {'thetae_advec': 'Theta-e advection',
                 'moist_advec': 'Moisture advection',
                 'qvectors': 'Q-vectors & divergence',
                 'vort_advec': 'Vort. advection'}
    
    cmaps = {'thetae_advec': 'bwr', 'moist_advec': 'BrBG', 'qvectors': 'bwr',
            'vort_advec': 'bwr'}
    
    kwargsdict = plot_kwargs(pres1, plot_type)
    kwargsdict['cmap'] = cmaps[plot_type]
    
    timestamps = coords_dict['time']
    lon2d, lat2d = np.meshgrid(coords_dict['lons'], coords_dict['lats'])
    sfcpres = sfc_dict['msl']
    hghts = pres_dict[pres1]['z']
    uwind = pres_dict[pres2]['u']
    vwind = pres_dict[pres2]['v']
    dx, dy = mpcalc.lat_lon_grid_deltas(coords_dict['lons'], coords_dict['lats'])
    if plot_type == 'vort_advec':
        advec = calc_dict[f'calc_{plot_type}'](dx, dy, lat2d, pres_dict, pres2)
    else:
        advec = calc_dict[f'calc_{plot_type}'](dx, dy, pres_dict, pres2)
    
    research_image_fold = '/Users/steiner/Documents/Python_plots/storm_composite_charts/'
    if os.path.exists(research_image_fold):
        print('Path exists!')
    else:
        os.mkdir(research_image_fold)
    cpressures = []
    for time_step in timerange:
    
        fig, ax = plot_backgroundmap(kwargsdict['plot_proj'], kwargsdict['scale'], kwargsdict['extent'])
        height_cons = ax.contour(lon2d, lat2d, hghts[time_step],
                                 levels=kwargsdict['levels_height'], linewidths=2.5,
                                  colors='black', transform=kwargsdict['data_proj'])
        
        sm_mslp_prev = ndimage.gaussian_filter(sfcpres[time_step-3], sigma=2)
        sm_mslp = ndimage.gaussian_filter(sfcpres[time_step], sigma=2)

        isallobars = ax.contour(lon2d, lat2d, (sm_mslp_prev - sm_mslp), levels=kwargsdict['levels_isall'], 
                             linewidths=1.5, colors='tab:red', transform=kwargsdict['data_proj'])
    
        isobars = ax.contour(lon2d, lat2d, sm_mslp, levels=kwargsdict['isobar_levels'],
                        linewidths=1.5, linestyles='--', colors='black', transform=kwargsdict['data_proj'])
        
        if plot_type != 'div':
            ax, advec_cons, units = plot_dict[f'plot_{plot_type}'](ax, lon2d, lat2d, advec[time_step], kwargsdict)
            plt.colorbar(advec_cons, orientation='horizontal', 
                     label=f'{title_dict[plot_type]} ({units})', pad=0.02, fraction=0.04)
        else:
            pass
    
        ax.barbs(lon2d[kwargsdict['wind_slice']], lat2d[kwargsdict['wind_slice']], 
                 uwind[time_step][kwargsdict['wind_slice']]*1.944,
                    vwind[time_step][kwargsdict['wind_slice']]*1.944,length=6, zorder=2, transform=kwargsdict['data_proj'])
    
        txt_timest = f"{timestamps[time_step].strftime('%Y-%m-%d %H')} UTC"
        contour_list = [height_cons, isobars, isallobars]
        for contour in contour_list:
            ax.clabel(contour, fontsize=kwargsdict['fontsize'], fmt='%d')
        cpressures.append(sfcpres[time_step].min())
        ax.set_title(txt_timest, fontsize=kwargsdict['fontsize'], fontweight='bold', loc='right')
        ax.set_title(f'ERA5 {pres1} MB height, {pres2} MB {title_dict[plot_type]}/wind, MSLP/pressure tendency (central_pressure = {sfcpres[time_step].min():.0f} MB)',loc='left', fontsize=11, fontweight='bold')
        fig.tight_layout()
        fig.savefig(f'{research_image_fold}{plot_type}_{txt_timest}.png', dpi=800)
        plt.show()
    return cpressures

def pres_data(data_dict):
    
    levels = list(data_dict['coords']['levels'])
    presVardict = {level: {k: v[:,i,:,:] for k, v in data_dict['data'].items()} for i, level in enumerate(levels)}
    return presVardict

midwest_upperfile = '/Users/steiner/Documents/reanalysis/1998_11upper.nc'
midwest_sfcfile = '/Users/steiner/Documents/reanalysis/1998_11sfc.nc'

eastcoast_stormfile_upper = '/Users/steiner/Documents/reanalysis/2018_01upper.nc'
eastcoast_sfcstormfile = '/Users/steiner/Documents/reanalysis/2018_01sfc.nc'

midwest_file_dict = {'sfc': midwest_sfcfile, 'upper': midwest_upperfile}
midwest_mfiles = {key: ReanalysisData(file) for key, file in midwest_file_dict.items()}

eastcoast_file_dict = {'sfc': eastcoast_sfcstormfile, 'upper': eastcoast_stormfile_upper}
eastmapping_files = {key: ReanalysisData(file) for key, file in eastcoast_file_dict.items()}

midwest_vars_dict = {key: val.formatting_data() for key, val in midwest_mfiles.items()}
eastcoast_vars_dict = {key: val.formatting_data() for key, val in eastmapping_files.items()}

pres_vardictM = pres_data(midwest_vars_dict['upper'])
pres_vardictE = pres_data(eastcoast_vars_dict['upper'])

time_range = range(18, 72, 3)
midwest_kwargs = dict(timerange = time_range, coords_dict=midwest_vars_dict['upper']['coords'],
                     sfc_dict=midwest_vars_dict['sfc']['data'], pres_dict=pres_vardictM, pres1=500, pres2=500,
                     plot_type='vort_advec')

eastcoast_kwargs = dict(timerange=time_range, coords_dict=eastcoast_vars_dict['upper']['coords'],
                       sfc_dict=eastcoast_vars_dict['sfc']['data'], pres_dict=pres_vardictE, pres1=500, pres2=500,
                       plot_type='vort_advec')

cmdwst = plot_composite_chart(**midwest_kwargs)
ceastcoast= plot_composite_chart(**eastcoast_kwargs)
