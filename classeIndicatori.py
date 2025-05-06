import xarray as xr
import numpy as np
import os
import xclim
from xclim.indices.stats import frequency_analysis
from xclim.core.calendar import percentile_doy
from xclim import units
from xclim.indices.stats import frequency_analysis
import pandas as pd
import cftime



"""
la classe si aspetta i seguenti parametri:
outputPath = (str) cartella dove salvare gli indicatori calcolati
annoInizio = (int) Anno inizio periodo storico
annoInizioFuturo = (int) Anno inizio periodo futuro
annoFine = (int) Anno fine periodo storico campo non obbligatorio se non valorizzato o =0 la funzione calcola automaticamente 30 anni dall'anno di inizio storico
annoFineFuturo = (int) Anno fine periodo futuro campo non obbligatorio se non valorizzato o =0 la funzione calcola automaticamente 30 anni dall'anno di fine futuro
modelData = (str) dataset da utilizzare al momento VHR-PRO-IT
scenario = (str) scenario da calcolare
path_variabili = (str) path dove recuperare le variabili 
calcola_anomalia = (ENUM) il valore di default è Y se il valore viene impostato a N non calcola l'anomalia sia assoluta che percentuale
"""
class Indicatori:
    def __init__(self, outputPath, annoInizio, annoInizioFuturo, annoFine=0, annoFineFuturo=0, modelData='VHR-PRO-IT',
                 scenario='RCP 4.5', path_variabili='/work/cmcc/ab03024/indicatori/VHR-PRO-IT', calcola_anomalia='Y'):
        self.outputPath = outputPath
        self.annoInizio = int(annoInizio)
        self.annoFine = int(annoFine)
        self.annoInizioFuturo = int(annoInizioFuturo)
        self.annoFineFuturo = int(annoFineFuturo)
        self.modelData = modelData
        self.scenario = scenario
        self.path_variabili = path_variabili
        self.path_indicatori_cache = outputPath + '/indicatori_1981-2070'
        self.scenario_dir = self.scenario.replace(' ', '').replace('.', '')
        self.calcola_anomalia = calcola_anomalia

        self.crea_cartella_se_non_esiste(self.path_indicatori_cache)

        if (self.annoFine == 0):
            self.annoFine = self.annoInizio + 30 - 1

        if (self.annoFineFuturo == 0):
            self.annoFineFuturo = self.annoInizioFuturo + 30 - 1

    def resample(self, ds, var):
        if var=='TOT_PREC':
            ds = ds.resample(time='1D').sum(dim='time')
        elif var=='TMAX_2M' or var=='U_10M' or var=='V_10M':
            ds = ds.resample(time='1D').max(dim='time')
        elif var=='TMIN_2M':
            ds = ds.resample(time='1D').min(dim='time')
        else:
            ds = ds.resample(time='1D').mean(dim='time')
        return ds

    def getNetcdfModelData(self, variabile='TMAX'):
        """
        Retrieve NetCDF model data for a specific variable.
        
        Args:
            variabile (str, optional): Variable name. Defaults to 'TMAX'.
        
        Returns:
            xarray.DataArray or None: Processed data array or None if model/variable not found
        """
        # Variable mapping
        variable_mapping = {
            'TMAX': ('TMAX_2M', 'degK', 'air_temperature', 'Daily maximum near-surface air temperature'),
            'TMAX_2M': ('TMAX_2M', 'degK', 'air_temperature', 'Daily maximum near-surface air temperature'),
            'T_2M': ('T_2M', 'degK', 'air_temperature', 'Daily minimum near-surface air temperature'),
            'TMIN': ('TMIN_2M', 'degK', 'air_temperature', 'Daily mean near-surface air temperature'),
            'TMIN_2M': ('TMIN_2M', 'degK', 'air_temperature', 'Daily mean near-surface air temperature'),
            'TOT_PREC': ('TOT_PREC', 'mm/day', 'precipitation amount', 'total precipitation amount'),
            'WIND': ('WIND', 'm/s', 'maximum daily wind speed', 'maximum daily wind speed')
        }
        
        # Check if variable is in mapping
        if variabile not in variable_mapping:
            print(f'Variabile non trovata {variabile}')
            return None
        
        # Unpack variable details
        variabile, units, standard_name, long_name = variable_mapping[variabile]
        
        # Prepare scenario
        scenario = self.scenario.replace(' ', '').replace('.', '')
        
        # Check model type
        if self.modelData == "VHR-PRO-IT":
            try:
                # Construct file path
                file_path = f'{self.path_variabili}/{variabile}_VHR-PRO-IT_1981_2070_{scenario}.nc'
                print(file_path)
                
                # Open dataset
                dsdata = xr.open_dataset(file_path)
                
                # Extract variable
                dsdata = dsdata[variabile]
                
                # Resample data
                data = self.resample(dsdata, variabile)
                
                # Set attributes
                data.attrs["units"] = units
                data.attrs['standard_name'] = standard_name
                data.attrs['long_name'] = long_name
                
                return data
            
            except Exception as e:
                print(f"Error processing data: {e}")
                return None
        
        else:
            print(f"Modello non riconosciuto: {self.modelData}")
            return None

    def merge_netcdf_files(self, file_list, prefix, annoInizio, annoFine):
        """
        Unisce i file NetCDF creati in un unico file.

        Parameters:
        file_list (list): Lista dei percorsi dei file NetCDF da unire.
        prefix (str): Prefisso per il nome del file di output.
        annoInizio (int): Anno di inizio del range.
        annoFine (int): Anno di fine del range.
        """
        combined = xr.open_mfdataset(file_list, combine='nested', concat_dim='time').sortby('time')
        output_file = f"{self.outputPath}/{self.modelData}_{prefix}_{annoInizio}_{annoFine}.nc"
        combined.to_netcdf(output_file, mode='w', format="NETCDF4")
        print(f"File combinato salvato in: {output_file}")
        # Elimina i file singoli
        for file in file_list:
            os.remove(file)
            print(f"File eliminato: {file}")

    def crea_cartella_se_non_esiste(self, cartella, debug=None):
        if not os.path.exists(cartella):
            os.makedirs(cartella)
            if debug != None: print(f"Cartella creata: {cartella}")
            return True
        else:
            if debug != None:  print(f"La cartella esiste già: {cartella}")
            return False

    def crea_anomalia_abs(self, ds):
        if self.calcola_anomalia =='N': return ds
        # crea anomalia assoluta
        annoInizio = str(self.annoInizio) + '-01-01'
        annoFine = str(self.annoFine) + '-12-31'
        annoInizioFuturo = str(self.annoInizioFuturo) + '-01-01'
        annoFineFuturo = str(self.annoFineFuturo) + '-12-31'

        passato = ds.sel(time=slice(annoInizio, annoFine)).mean(dim='time')
        futuro = ds.sel(time=slice(annoInizioFuturo, annoFineFuturo)).mean(dim='time')

        anomalia = futuro - passato
        return anomalia

    def crea_anomalia_percentuale(self, ds):
        if self.calcola_anomalia =='N': return ds
        # crea anomalia assoluta
        annoInizio = str(self.annoInizio) + '-01-01'
        annoFine = str(self.annoFine) + '-12-31'
        annoInizioFuturo = str(self.annoInizioFuturo) + '-01-01'
        annoFineFuturo = str(self.annoFineFuturo) + '-12-31'

        passato = ds.sel(time=slice(annoInizio, annoFine)).mean(dim='time')
        futuro = ds.sel(time=slice(annoInizioFuturo, annoFineFuturo)).mean(dim='time')

        anomalia = (futuro-passato) / passato
        return anomalia * 100

    def calcolaPercentile(self, variabile, window=5, percentile=10, annoInizio='0', annoFine='0'):
        """
        Calcolo il percentile con la libreria di xclim se non lo trova tra quelli già calcolati

        """
        ref_start = f"{self.annoInizio}-{self.annoFine}"
        

        path_percentile = os.path.join(self.outputPath,
                                   f"{variabile.lower()}_{percentile}th-percentile_window-{window}_{ref_start}.nc")
        if os.path.isfile(path_percentile):
            dspercentile = xr.open_dataset(path_percentile)
        else:

            print(f"Calcolo del {percentile}° percentile di {variabile} con finestra di {window} giorni per il periodo {ref_start}")

            ds = self.getNetcdfModelData(variabile)
            if(annoInizio=='0'): annoInizio = str(self.annoInizio) + '-01-01'
            if(annoFine=='0'): annoFine = str(self.annoFine) + '-12-31'
            
            periodo_passato = ds.sel(time=slice(annoInizio, annoFine))


            dspercentile = percentile_doy(periodo_passato, window=window, per=percentile).sel(
                percentiles=percentile)
            
            dspercentile = dspercentile.assign_coords({'lon' : ds.lon, 'lat':ds.lat})


            # Salva il risultato in un file NetCDF

            dspercentile.to_netcdf(path_percentile, mode='w', format="NETCDF4")

            print(f"File salvato: {path_percentile}")

        return dspercentile['per']
    
    def calcolo_spi(self, data, data_ref, scale = 3, nseas= 12):
        """
        Calcola l'Indice di Precipitazione Standardizzato (SPI)

        Parametri:
        data (xarray.DataArray): Dati mensili per il periodo di calcolo
        data_ref (xarray.DataArray): Dati mensili per il periodo di riferimento
        scale (int): Scala di accumulo (1, 3, 12, 48)
        nseas (int): Numero di stagioni (mensile = 12)

        Ritorna:
        xarray.DataArray: SPI calcolato
        """
        
        def prepare_data(da, scale):
            # Prepara i dati per il calcolo SPI
            rolled = xr.concat([da.roll(time=-i) for i in range(scale)], dim='scale')
            summed = rolled.sum('scale')
            return summed.dropna('time')

        # Prepara i dati
        XS = prepare_data(data, scale)
        XSref = prepare_data(data_ref, scale)

        # Rimuovi i primi mesi se necessario
        erase_yr = np.ceil(scale / 12)
        if scale > 1:
            XS = XS.isel(time=slice(int(nseas * erase_yr - scale + 1), None))
            XSref = XSref.isel(time=slice(int(nseas * erase_yr - scale + 1), None))

        def calculate_spi(group):
            month = group.time.dt.month.values[0]
            x = group.values
            xref = XSref.sel(time=XSref.time.dt.month == month).values

            # Rimuovi i NaN e gli zeri
            xref_nozero = xref[xref > 0]
            q = np.sum(xref == 0) / len(xref)

            # Calcola i parametri della distribuzione gamma
            shape, loc, scale = stats.gamma.fit(xref_nozero, floc=0)

            # Calcola la CDF
            cdf = q + (1 - q) * stats.gamma.cdf(x, shape, loc, scale)

            # Converti in SPI
            return stats.norm.ppf(cdf)

        # Applica il calcolo SPI
        spi = XS.groupby('time.month').apply(calculate_spi)

        return spi

    # alebo
    def cd(self, indicatore='CD', freq='YS'): 
        """
        CD-Giorni freschi-secchi (giorni)
        
        Numero di giorni con temperatura media giornaliera minore del 25° percentile** della temperatura media giornaliera 
        e con precipitazione giornaliera minore del 25° percentile** della precipitazione giornaliera.
        """
        scenario_dir = self.scenario.replace(' ', '').replace('.', '')
        file_indicatore_completo = f'{self.path_indicatori_cache}/{indicatore.lower()}_1980-2070_{scenario_dir}.nc'
        
        if os.path.isfile(file_indicatore_completo):
            dataset = xr.open_dataset(file_indicatore_completo)
        else:

            # Calcolo il 25° percentiles con calcolaPercentile
            tmean_25p = self.calcolaPercentile('T_2M', window=5, percentile=25)
            prec_25p = self.calcolaPercentile('TOT_PREC', window=5, percentile=25)

            # Temperatura media giornaliera
            tmean = self.getNetcdfModelData('T_2M')

            # Precipitazione giornaliera
            prec = self.getNetcdfModelData('TOT_PREC')
            
            # Trovo i Gioni freschi-secchi
            cold_dry = xclim.indices.cold_and_dry_days(tmean, prec, tmean_25p, prec_25p, freq=freq)
            
            # creo il dataset espicitando tempo e dimensioni
            dataset = xr.Dataset({
                indicatore: (['time', 'rlat', 'rlon'], cold_dry.values),
                'time': cold_dry.time,
                'rlat': cold_dry.rlat,
                'rlon': cold_dry.rlon
            })
            
            # aggiungo le coordinate
            dataset = dataset.assign_coords({'lon': cold_dry.lon, 'lat': cold_dry.lat})
            
            # salvo il file per l'intero periodo
            dataset.to_netcdf(file_indicatore_completo, mode='w', format="NETCDF4",
                            encoding={'time': {'dtype': 'int32', 'units': 'days since 1900-01-01'}})
        
        # Calcolo anomalia assoluta
        anomalia = self.crea_anomalia_abs(dataset)
        
        # Salvo anomalia
        output_file = f'{self.outputPath}/{self.modelData}_{indicatore.lower()}_{self.annoInizio}-{self.annoFine}_{self.annoInizioFuturo}-{self.annoFineFuturo}_{scenario_dir}.nc'
        anomalia.to_netcdf(output_file, mode='w', format="NETCDF4")
        
        return anomalia
    
    def cdd(self, indicatore='CDD', thresh='1 mm/day', freq='YS'):
        # thresh = thresh + ' mm/day'
        # Alebo
        # xclim.indices.maximum_consecutive_dry_days(pr, thresh='1 mm/day', freq='YS', resample_before_rl=True)
        scenario_dir = self.scenario.replace(' ', '').replace('.', '')

        # cambia il nome con passato e futuro
        file_indicatore_completo = f'{self.path_indicatori_cache}/{indicatore.lower()}_1980-2070_{scenario_dir}.nc'
        if os.path.isfile(file_indicatore_completo):
            cdd_dataset = xr.open_dataset(file_indicatore_completo)
        else:
            pr = self.getNetcdfModelData('TOT_PREC')

            # setto a 0 il rumore del modello
            pr = xr.where(pr < 1, 0, pr)

            # combined_ds['TOT_PREC'] = combined_ds['TOT_PREC'].assign_attrs(units='mm/day')
            # pr['TOT_PREC'].attrs['units'] = 'mm/day'
            pr = pr.assign_attrs(units='mm/day')
            indicatore_tot = xclim.indices.maximum_consecutive_dry_days(pr, thresh=thresh, freq=freq,
                                                                        resample_before_rl=True)

            cdd_dataset = xr.Dataset({
                indicatore: (['time', 'rlat', 'rlon'], indicatore_tot.values),
                'time': indicatore_tot.time,  # Usa l'intero oggetto time, non solo l'anno
                'rlat': indicatore_tot.rlat,
                'rlon': indicatore_tot.rlon
            })

            # aggiungo le coordinate
            cdd_dataset = cdd_dataset.assign_coords({'lon': indicatore_tot.lon, 'lat': indicatore_tot.lat})

            # Aggiungi gli attributi
            # cdd_dataset[indicatore].attrs['units'] = 'mm/day'
            '''
            cdd_dataset[indicatore].attrs[
                'long_name'] = 'On a summer day, the maximum temperature is ' + thresh + '. The number of summer days per year is a frequently used climate indicator.'
            cdd_dataset[indicatore].attrs['standard_name'] = 'Summer Days ' + thresh
            cdd_dataset.attrs['reference'] = "https://etccdi.pacificclimate.org/list_27_indices.shtml"
            '''

            cdd_dataset.to_netcdf(file_indicatore_completo, mode='w', format="NETCDF4",
                                  encoding={'time': {'dtype': 'int32', 'units': 'days since 1900-01-01'}})

        # print(cdd_dataset)
        anomalia = self.crea_anomalia_abs(cdd_dataset)

        output_file = f'{self.outputPath}/{self.modelData}_{indicatore.lower()}_{self.annoInizio}-{self.annoFine}_{self.annoInizioFuturo}-{self.annoFineFuturo}_{scenario_dir}.nc'
        anomalia.to_netcdf(output_file, mode='w', format="NETCDF4")
        return anomalia

    def cdds(self, indicatore = 'CDDS', thresh='21 degC', freq='YS'):
        """
        Paolo
        cdds definizione PNACC: somma della temperatura media giornaliera meno 21°C se la temperatura media giornaliera è maggiore di 24°C
        xclim.indices.cooling_degree_days: Ritorna la somma delle differenze tra le temperature medie e il threshold se e solo se la temperatura media è superiore al threshold 
        """
        scenario = self.scenario.replace(' ', '').replace('.', '')
        file_completo =  f'{self.path_indicatori_cache}/{indicatore.lower()}_{self.annoInizio}-{self.annoFine}_{self.annoInizioFuturo}-{self.annoFineFuturo}_{scenario}.nc'
        if os.path.isfile(file_completo):
            cdds = xr.open_dataset(file_completo)
        else:
            ds_tas = self.getNetcdfModelData(variabile='T_2M')
            ds_tas = ds_tas - 273.15
            ds_tas.attrs['units'] = 'degC'
            # filtro il dataset mantenendo solo le temperature sopra 24°C (condizione definizione PNACC)
            mask = ds_tas >= 24
            ds_tas = ds_tas.where(mask, 0)
            # applico funzione xclim con threshold a 21°C. Quindi ccds_raw contiene la differenza fra tas e threshold (21°C) dove tas sono unicamente i valori > 24°C
            cdds_raw = xclim.indices.cooling_degree_days(tas=ds_tas, thresh=thresh, freq=freq)
            cdds = xr.Dataset({indicatore: (['time', 'rlat', 'rlon'], cdds_raw.values),
                'time': cdds_raw.time,  # Usa l'intero oggetto time, non solo l'anno
                'rlat': cdds_raw.rlat,
                'rlon': cdds_raw.rlon})
            
            cdds = cdds.assign_coords({'lon': cdds_raw.lon, 'lat': cdds_raw.lat})
            cdds.to_netcdf(file_completo, mode='w', format='NETCDF4', encoding={'time': {'dtype': 'int32', 'units': 'days since 1900-01-01'}})
        
        cdds_abs_anomaly = self.crea_anomalia_abs(cdds)
        output_file = f'{self.outputPath}/{self.modelData}_{indicatore.lower()}_{self.annoInizio}-{self.annoFine}_{self.annoInizioFuturo}-{self.annoFineFuturo}_{scenario}.nc'
        cdds_abs_anomaly.to_netcdf(output_file, mode='w', format="NETCDF4")
        return cdds_abs_anomaly  
        
    def cfd(self, indicatore='CFD', thresh='0.0 degC', freq='YS-JUL', resample_before_rl=True): #Paolo
        scenario = self.scenario.replace(' ', '').replace('.', '')
        file_completo = f'{self.path_indicatori_cache}/{indicatore.lower()}_1980-2070_{scenario}.nc'
        if os.path.isfile(file_completo):
            cfd = xr.open_dataset(file_completo)
        else:
            ds_tasmin = self.getNetcdfModelData(variabile = 'TMIN_2M')
            ds_tasmin = ds_tasmin - 273.15
            ds_tasmin.attrs['units'] = 'degC'
            cfd_raw = xclim.indices.maximum_consecutive_frost_days(tasmin=ds_tasmin, thresh=thresh, freq=freq, resample_before_rl=True)
            cfd = xr.Dataset({
                indicatore: (['time', 'rlat', 'rlon'], cfd_raw.values),
                'time': cfd_raw.time,  # Usa l'intero oggetto time, non solo l'anno
                'rlat': cfd_raw.rlat,
                'rlon': cfd_raw.rlon})
            cfd = cfd.assign_coords({'lon': cfd_raw.lon, 'lat': cfd_raw.lat})
            cfd.to_netcdf(file_completo, mode='w', format='NETCDF4', encoding={'time': {'dtype': 'int32', 'units': 'days since 1900-01-01'}})      

        cfd_abs_anomaly = self.crea_anomalia_abs(cfd)
        output_file = f'{self.outputPath}/{self.modelData}_{indicatore.lower()}_{self.annoInizio}-{self.annoFine}_{self.annoInizioFuturo}-{self.annoFineFuturo}_{scenario}.nc'
        cfd_abs_anomaly.to_netcdf(output_file, mode='w', format="NETCDF4")
        return cfd_abs_anomaly

    def csdi(self, indicatore = 'CSDI', window = 6, freq = 'YS', op = '<'):
        
        scenario_dir = self.scenario.replace(' ', '').replace('.', '')
        #tasmin_per_ds = self.calcolaPercentile('TMIN_2M', window=5, percentile=10)
        tasmin_per_ds = xr.open_dataset('/work/cmcc/gg21021/PNRR_NBFC/indicatori/tmin_2m_10th-percentile_window-5_1981-2010.nc')

        tasmin_per = tasmin_per_ds['per'] - 273.15
        tasmin_per.attrs['units'] = 'degC'
        
        file_indicatore_completo = f'{self.path_indicatori_cache}/{indicatore.lower()}_{self.annoInizio}-{self.annoFine}_{self.annoInizioFuturo}-{self.annoFineFuturo}_{scenario_dir}.nc'
        
        
        if os.path.isfile(file_indicatore_completo):
            dataset = xr.open_dataset(file_indicatore_completo)
        else:
            tasmin = self.getNetcdfModelData('TMIN_2M')

        
        
            indicatore_tot = xclim.indices.cold_spell_duration_index(tasmin, tasmin_per, window=window, freq=freq, op=op)
            
            dataset = xr.Dataset({
                indicatore: (['time', 'rlat', 'rlon'], indicatore_tot.values),
                'time': indicatore_tot.time,  # Usa l'intero oggetto time, non solo l'anno
                'rlat': indicatore_tot.rlat,
                'rlon': indicatore_tot.rlon
            })

            # aggiungo le coordinate
            dataset = dataset.assign_coords({'lon': indicatore_tot.lon, 'lat': indicatore_tot.lat})


            dataset.to_netcdf(file_indicatore_completo, mode='w', format="NETCDF4",
                                  encoding={'time': {'dtype': 'int32', 'units': 'days since 1900-01-01'}})

        # print(cdd_dataset)
        anomalia = self.crea_anomalia_abs(dataset)

        output_file = f'{self.outputPath}/{self.modelData}_{indicatore.lower()}_{self.annoInizio}-{self.annoFine}_{self.annoInizioFuturo}-{self.annoFineFuturo}_{scenario_dir}.nc'
        anomalia.to_netcdf(output_file, mode='w', format="NETCDF4")
        return anomalia

    # alebo
    def cw(self, indicatore='CW', freq='YS'):
        """
        CW-Giorni freschi-piovosi (giorni)
        
        Numero di giorni con temperatura media giornaliera minore del 25° percentile** della temperatura media giornaliera 
        e con precipitazione giornaliera maggiore del 75° percentile** della precipitazione giornaliera.
        """
        file_indicatore_completo = f'{self.path_indicatori_cache}/{indicatore.lower()}_1980-2070_{self.scenario_dir}.nc'
        
        if os.path.isfile(file_indicatore_completo):
            dataset = xr.open_dataset(file_indicatore_completo)
        else:
            # Calcolo il 75° percentiles con calcolaPercentile
            tmean_25p = self.calcolaPercentile('T_2M', window=5, percentile=25)
            prec_75p = self.calcolaPercentile('TOT_PREC', window=5, percentile=75)

            # Temperatura media giornaliera
            tmean = self.getNetcdfModelData('T_2M')

            # Precipitazione giornaliera
            prec = self.getNetcdfModelData('TOT_PREC')
            
            # Conversione temperature in Celsius (forse inutile)
            tmean_25p = tmean_25p - 273.15
            tmean_25p.attrs['units'] = 'degC'

            # Trovo i Gioni caldi-secchi
            cold_wet = xclim.indices.cold_and_wet_days(tmean, prec, tmean_25p, prec_75p, freq=freq)
            
            # Conto i Gioni caldi-secchi per [freq]
            cw_count = cold_wet.resample(time=freq).sum(dim='time')
            
            # creo il dataset espicitando tempo e dimensioni
            dataset = xr.Dataset({
                indicatore: (['time', 'rlat', 'rlon'], cw_count.values),
                'time': cw_count.time,
                'rlat': cw_count.rlat,
                'rlon': cw_count.rlon
            })
            
            # aggiungo le coordinate
            dataset = dataset.assign_coords({'lon': cw_count.lon, 'lat': cw_count.lat})
            
            # salvo il file per l'intero periodo
            dataset.to_netcdf(file_indicatore_completo, mode='w', format="NETCDF4",
                            encoding={'time': {'dtype': 'int32', 'units': 'days since 1900-01-01'}})
        
        # Calcolo anomalia assoluta
        anomalia = self.crea_anomalia_abs(dataset)
        
        # Salvo anomalia
        output_file = f'{self.outputPath}/{self.modelData}_{indicatore.lower()}_{self.annoInizio}-{self.annoFine}_{self.annoInizioFuturo}-{self.annoFineFuturo}_{self.scenario_dir}.nc'
        anomalia.to_netcdf(output_file, mode='w', format="NETCDF4")
        
        return anomalia

    def ews(self, indicatore='EWS'):
        # Paolo - 98esimo percentile della massima velocità giornaliera del vento 
        scenario = self.scenario.replace(' ', '').replace('.', '')
        file_passato = f'{self.path_indicatori_cache}/{indicatore.lower()}_percentili_{self.annoInizio}-{self.annoFine}.nc'
        file_futuro = f'{self.path_indicatori_cache}/{indicatore.lower()}_percentili_{self.annoInizioFuturo}-{self.annoFineFuturo}.nc'
        output_file = f'{self.outputPath}/{self.modelData}_{indicatore.lower()}_{self.annoInizio}-{self.annoFine}_{self.annoInizioFuturo}-{self.annoFineFuturo}_{scenario}.nc'
        #controllo se il file con l'anomalia già esiste
        if os.path.isfile(output_file):
            ews_anomalia_percentuale = xr.open_dataset(output_file)
        else:
            #controllo se il file con il percentile calcolato sullo storico esiste       
            if os.path.isfile(file_passato):
                perc_wind_passato = xr.open_dataset(file_passato)
                ds_wind_futuro = (self.getNetcdfModelData(variabile='WIND')).sel(time=slice(str(self.annoInizioFuturo) + '-01-01', str(self.annoFineFuturo) + '-12-31')) 
            else:
                ds_wind = self.getNetcdfModelData(variabile='WIND')
                ds_wind_passato = ds_wind.sel(time=slice(str(self.annoInizio) + '-01-01', str(self.annoFine) + '-12-31'))
                perc_wind_passato = ds_wind_passato.quantile(q=0.98, dim='time')
                perc_wind_passato.to_netcdf(file_passato, mode='w', format='NETCDF4')
                ds_wind_futuro = ds_wind.sel(time=slice(str(self.annoInizioFuturo) + '-01-01', str(self.annoFineFuturo) + '-12-31'))
            
            perc_wind_futuro = ds_wind_futuro.quantile(q=0.98, dim='time')
            perc_wind_futuro.to_netcdf(file_futuro, mode='w', format='NETCDF4')
            
            #calcolo l'anomalia percentuale
            ews_anomalia_percentuale_raw = (perc_wind_futuro - perc_wind_passato) / perc_wind_passato * 100
            ews_anomalia_percentuale = ews_anomalia_percentuale_raw.rename_vars({'WIND': indicatore})
            ews_anomalia_percentuale.to_netcdf(output_file, mode='w', format='NETCDF4')

        return ews_anomalia_percentuale

    def fd(self, freq='1Y', indicatore='FD'):
        scenario_dir = self.scenario.replace(' ', '').replace('.', '')

        # cambia il nome con passato e futuro
        file_indicatore_completo = f'{self.path_indicatori_cache}/{indicatore.lower()}_1980-2070_{scenario_dir}.nc'
        if os.path.isfile(file_indicatore_completo):
            dataset = xr.open_dataset(file_indicatore_completo)
        else:
            tmin = self.getNetcdfModelData('TMIN_2M')

            # setto a 0 il rumore del modello
            fd= xr.where(tmin <= 0 + 273.15, 0, 1)

            indicatore_tot = fd.resample(time=freq).sum(dim='time')

            dataset = xr.Dataset({
                indicatore: (['time', 'rlat', 'rlon'], indicatore_tot.values),
                'time': indicatore_tot.time,  # Usa l'intero oggetto time, non solo l'anno
                'rlat': indicatore_tot.rlat,
                'rlon': indicatore_tot.rlon
            })
            # aggiungo le coordinate
            dataset = dataset.assign_coords({'lon': indicatore_tot.lon, 'lat': indicatore_tot.lat})

            dataset.to_netcdf(file_indicatore_completo, mode='w', format="NETCDF4",
                              encoding={'time': {'dtype': 'int32', 'units': 'days since 1900-01-01'}})

        # print(cdd_dataset)
        anomalia = self.crea_anomalia_abs(dataset)

        output_file = f'{self.outputPath}/{self.modelData}_{indicatore.lower()}_{self.annoInizio}-{self.annoFine}_{self.annoInizioFuturo}-{self.annoFineFuturo}_{scenario_dir}.nc'
        anomalia.to_netcdf(output_file, mode='w', format="NETCDF4")
        return anomalia

    def humidex (self, indicatore='HUMIDEX5', freq='1Y', classe=0):
        scenario_dir = self.scenario.replace(' ', '').replace('.', '')
        file_indicatore_completo = f'{self.path_indicatori_cache}/{indicatore.lower()}_{self.annoInizio}-{self.annoFine}_{self.annoInizioFuturo}-{self.annoFineFuturo}_{scenario_dir}.nc'

        if os.path.isfile(file_indicatore_completo):
            dataset = xr.open_dataset(file_indicatore_completo)
        else:
            # Temperatura media giornaliera
            tasmax = self.getNetcdfModelData('TMAX_2M') 

            # Precipitazione giornaliera
            rh = self.getNetcdfModelData('HURS')
 
            #  Humidex
            humidex= xclim.indices.humidex(tasmax, hurs=rh)

            # trasformo in gradi da celsius
            humidex=humidex-273.15
          
            humidex_classes = {
                1: (None, 20),
                2: (20, 30),
                3: (30, 40),
                4: (40, 45),
                5: (45, None)
            }

            #indicatore_tot=xr.where(humidex >= thresh, 1, 0)
            indicatore_tot = indicatore_tot.resample(time = freq).sum(dim='time')

            dataset = xr.Dataset({
                indicatore: (['time', 'rlat', 'rlon'], indicatore_tot.values),
                'time': indicatore_tot.time,  # Usa l'intero oggetto time, non solo l'anno
                'rlat': indicatore_tot.rlat,
                'rlon': indicatore_tot.rlon
            })

            # aggiungo le coordinate
            dataset = dataset.assign_coords({'lon': indicatore_tot.lon, 'lat': indicatore_tot.lat})


            dataset.to_netcdf(file_indicatore_completo, mode='w', format="NETCDF4",
                                  encoding={'time': {'dtype': 'int32', 'units': 'days since 1900-01-01'}})

        # print(cdd_dataset)
        anomalia = self.crea_anomalia_abs(dataset)

        output_file = f'{self.outputPath}/{self.modelData}_{indicatore.lower()}_{self.annoInizio}-{self.annoFine}_{self.annoInizioFuturo}-{self.annoFineFuturo}_{self.scenario_dir}.nc'
        anomalia.to_netcdf(output_file, mode='w', format="NETCDF4")
        return anomalia
    
    def humidex1 (self):
        self.humidex(indicatore='HUMIDEX1', freq='1Y', classe=1)

    def humidex2 (self):
        self.humidex(indicatore='HUMIDEX2', freq='1Y', classe=2)

    def humidex3 (self):
        self.humidex(indicatore='HUMIDEX3', freq='1Y', classe=3)

    def humidex4 (self):
        self.humidex(indicatore='HUMIDEX4', freq='1Y', classe=4)

    def humidex5 (self):
        self.humidex(indicatore='HUMIDEX5', freq='1Y', classe=5)
        
    def hw(self, freq='1Y'):
        return self.su(thresh=35, freq=freq, indicatore='HW')

    def hw37(self, freq='1Y'):
        return self.su(thresh=37, freq=freq, indicatore='HW37')

    def pet(self, indicatore='PET'):
        scenario = self.scenario.replace(' ', '').replace('.', '')
        file_indicatore_completo = f'{self.path_indicatori_cache}/{indicatore.lower()}_{self.annoInizio}-{self.annoFineFuturo}_{scenario}.nc'

        if os.path.isfile(file_indicatore_completo):
            ds_pet = xr.open_dataset(file_indicatore_completo)
        else:
            ds_tas = self.getNetcdfModelData(variabile='T_2M')
            pet_raw = xclim.indices.potential_evapotranspiration(method='TW48', tas = ds_tas)

            pet_raw_transformed = pet_raw * 2630000 #Cambiare l'unità da kg m-2 s-1 a mm/month

            ds_pet = xr.Dataset({
                indicatore: (['time', 'rlat', 'rlon'], pet_raw_transformed.values),
                'time': pet_raw_transformed.time,
                'rlat': pet_raw_transformed.rlat,
                'rlon': pet_raw_transformed.rlon
            })

            ds_pet.to_netcdf(file_indicatore_completo, mode='w', format='NETCDF4') #Il file salvato ha un intervallo temporale mensile ma il valore è in mm/day

        variazione = self.crea_anomalia_percentuale(ds_pet)
        output_file = f'{self.outputPath}/{self.modelData}_{indicatore.lower()}_{self.annoInizio}-{self.annoFine}_{self.annoInizioFuturo}-{self.annoFineFuturo}_{scenario}.nc'
        variazione.to_netcdf(output_file, mode='w', format='NETCDF4')
        return variazione
    
    def pr95prctile(self, indicatore='PR95PRCTILE'):
        
        file_indicatore_completo = f'{self.path_indicatori_cache}/{indicatore.lower()}_1980-2070_{self.scenario_dir}.nc'
        
        if os.path.isfile(file_indicatore_completo):
            dataset = xr.open_dataset(file_indicatore_completo)
        else:
            # Precipitazione giornaliera
            prec = self.getNetcdfModelData('TOT_PREC')
            
            # Calcolo il 95° percentiles 
            indicatore_tot=prec.quantile(0.95,dim='time', skipna=True)
            #indicatore_tot = pr95prctile.resample(time = freq).mean(dim='time')

            dataset = xr.Dataset({
                indicatore: (['time', 'rlat', 'rlon'], indicatore_tot.values),
                'time': indicatore_tot.time,  # Usa l'intero oggetto time, non solo l'anno
                'rlat': indicatore_tot.rlat,
                'rlon': indicatore_tot.rlon
            })

            # aggiungo le coordinate
            dataset = dataset.assign_coords({'lon': indicatore_tot.lon, 'lat': indicatore_tot.lat})


            dataset.to_netcdf(file_indicatore_completo, mode='w', format="NETCDF4",
                                  encoding={'time': {'dtype': 'int32', 'units': 'days since 1900-01-01'}})
       
        anomalia = self.crea_anomalia_percentuale(dataset)

        output_file = f'{self.outputPath}/{self.modelData}_{indicatore.lower()}_{self.annoInizio}-{self.annoFine}_{self.annoInizioFuturo}-{self.annoFineFuturo}_{self.scenario_dir}.nc'
        anomalia.to_netcdf(output_file, mode='w', format="NETCDF4")

    def pr95prctile(self, indicatore='PR95PRCTILE'):
        
        file_indicatore_completo = f'{self.path_indicatori_cache}/{indicatore.lower()}_1980-2070_{self.scenario_dir}.nc'
        
        if os.path.isfile(file_indicatore_completo):
            dataset = xr.open_dataset(file_indicatore_completo)
        else:
            # Precipitazione giornaliera
            prec = self.getNetcdfModelData('TOT_PREC')
            
            # Calcolo il 95° percentiles 
            indicatore_tot=prec.quantile(0.95,dim='time', skipna=True)
            #indicatore_tot = pr95prctile.resample(time = freq).mean(dim='time')

            dataset = xr.Dataset({
                indicatore: (['time', 'rlat', 'rlon'], indicatore_tot.values),
                'time': indicatore_tot.time,  # Usa l'intero oggetto time, non solo l'anno
                'rlat': indicatore_tot.rlat,
                'rlon': indicatore_tot.rlon
            })

            # aggiungo le coordinate
            dataset = dataset.assign_coords({'lon': indicatore_tot.lon, 'lat': indicatore_tot.lat})


            dataset.to_netcdf(file_indicatore_completo, mode='w', format="NETCDF4",
                                  encoding={'time': {'dtype': 'int32', 'units': 'days since 1900-01-01'}})
       
        anomalia = self.crea_anomalia_percentuale(dataset)

        output_file = f'{self.outputPath}/{self.modelData}_{indicatore.lower()}_{self.annoInizio}-{self.annoFine}_{self.annoInizioFuturo}-{self.annoFineFuturo}_{self.scenario_dir}.nc'
        anomalia.to_netcdf(output_file, mode='w', format="NETCDF4")

    def pr99prctile(self, indicatore='PR99PRCTILE'):
        
        file_indicatore_completo = f'{self.path_indicatori_cache}/{indicatore.lower()}_1980-2070_{self.scenario_dir}.nc'
        
        if os.path.isfile(file_indicatore_completo):
            dataset = xr.open_dataset(file_indicatore_completo)
        else:
            # Precipitazione giornaliera
            prec = self.getNetcdfModelData('TOT_PREC')
            
            # Calcolo il 99° percentiles 
            indicatore_tot=prec.quantile(0.99,dim='time', skipna=True)
            #indicatore_tot = pr95prctile.resample(time = freq).mean(dim='time')

            dataset = xr.Dataset({
                indicatore: (['time', 'rlat', 'rlon'], indicatore_tot.values),
                'time': indicatore_tot.time,  # Usa l'intero oggetto time, non solo l'anno
                'rlat': indicatore_tot.rlat,
                'rlon': indicatore_tot.rlon
            })

            # aggiungo le coordinate
            dataset = dataset.assign_coords({'lon': indicatore_tot.lon, 'lat': indicatore_tot.lat})


            dataset.to_netcdf(file_indicatore_completo, mode='w', format="NETCDF4",
                                  encoding={'time': {'dtype': 'int32', 'units': 'days since 1900-01-01'}})
       
        anomalia = self.crea_anomalia_percentuale(dataset)

        output_file = f'{self.outputPath}/{self.modelData}_{indicatore.lower()}_{self.annoInizio}-{self.annoFine}_{self.annoInizioFuturo}-{self.annoFineFuturo}_{self.scenario_dir}.nc'
        anomalia.to_netcdf(output_file, mode='w', format="NETCDF4")

    
    def prcptot(self, freq='YS', indicatore='PRCPTOT'):
        print(f'Calcolo {indicatore}: Precipitazione cumulata nei giorni piovosi (>= 1 mm)')

        # cambia il nome con passato e futuro
        file_indicatore_completo = f'{self.path_indicatori_cache}/{indicatore.lower()}_1980-2070_{self.scenario_dir}.nc'
        if os.path.isfile(file_indicatore_completo):
            prcptot_dataset = xr.open_dataset(file_indicatore_completo)
        else:
            precip_data = self.getNetcdfModelData(variabile='TOT_PREC')

            # setto a 0 il rumore del modello
            precip_data = xr.where(precip_data < 1, 0, precip_data)

            precip_data = precip_data.assign_attrs(units='mm/day')
            prcptot_result = precip_data.where(precip_data >= 1).resample(time=freq).sum()

            prcptot_dataset = xr.Dataset({
                indicatore: (['time', 'rlat', 'rlon'], prcptot_result.values),
                'time': prcptot_result.time,  # Usa l'intero oggetto time, non solo l'anno
                'rlat': prcptot_result.rlat,
                'rlon': prcptot_result.rlon
            })

            # aggiungo le coordinate
            prcptot_dataset = prcptot_dataset.assign_coords({'lon': prcptot_result.lon, 'lat': prcptot_result.lat})

            # Aggiungi gli attributi
            # cdd_dataset[indicatore].attrs['units'] = 'mm/day'
            '''
            cdd_dataset[indicatore].attrs[
                'long_name'] = 'On a summer day, the maximum temperature is ' + thresh + '. The number of summer days per year is a frequently used climate indicator.'
            cdd_dataset[indicatore].attrs['standard_name'] = 'Summer Days ' + thresh
            cdd_dataset.attrs['reference'] = "https://etccdi.pacificclimate.org/list_27_indices.shtml"
            '''

            prcptot_dataset.to_netcdf(file_indicatore_completo, mode='w', format="NETCDF4",
                                  encoding={'time': {'dtype': 'int32', 'units': 'days since 1900-01-01'}})

        # print(cdd_dataset)
        anomalia = self.crea_anomalia_percentuale(prcptot_dataset)

        output_file = f'{self.outputPath}/{self.modelData}_{indicatore.lower()}_{self.annoInizio}-{self.annoFine}_{self.annoInizioFuturo}-{self.annoFineFuturo}_{self.scenario_dir}.nc'
        anomalia.to_netcdf(output_file, mode='w', format="NETCDF4")
        return anomalia
    
    def rx1day(self, indicatore='RX1DAY', thresh=1, freq='YS'):
        scenario_dir = self.scenario.replace(' ', '').replace('.', '')

        # cambia il nome con passato e futuro
        file_indicatore_completo = f'{self.path_indicatori_cache}/{indicatore.lower()}_1980-2070_{scenario_dir}.nc'
        if os.path.isfile(file_indicatore_completo):
            dataset = xr.open_dataset(file_indicatore_completo)
        else:
            pr = self.getNetcdfModelData('TOT_PREC')

            # setto a 0 il rumore del modello
            pr = xr.where(pr < 1, 0, pr)
            pr = pr.assign_attrs(units='mm/day')
            
            indicatore_tot = xclim.indices.max_n_day_precipitation_amount(pr, window=thresh, freq=freq)

            dataset = xr.Dataset({
                indicatore: (['time', 'rlat', 'rlon'], indicatore_tot.values),
                'time': indicatore_tot.time,  # Usa l'intero oggetto time, non solo l'anno
                'rlat': indicatore_tot.rlat,
                'rlon': indicatore_tot.rlon
            })

            # aggiungo le coordinate
            dataset = dataset.assign_coords({'lon': indicatore_tot.lon, 'lat': indicatore_tot.lat})


            dataset.to_netcdf(file_indicatore_completo, mode='w', format="NETCDF4",
                                  encoding={'time': {'dtype': 'int32', 'units': 'days since 1900-01-01'}})

        # print(cdd_dataset)
        anomalia = self.crea_anomalia_percentuale(dataset)

        output_file = f'{self.outputPath}/{self.modelData}_{indicatore.lower()}_{self.annoInizio}-{self.annoFine}_{self.annoInizioFuturo}-{self.annoFineFuturo}_{scenario_dir}.nc'
        anomalia.to_netcdf(output_file, mode='w', format="NETCDF4")
        return anomalia
    
    def rx5day(self):
        return self.rx1day(indicatore='RX5DAY', thresh=5, freq='YS')
    
    def rx3day(self):
        return self.rx1day(indicatore='RX3DAY', thresh=3, freq='YS')
    
    def r10(self, freq='1Y', thresh=10, indicatore='R10'):
        self.r20(freq='1Y', thresh=thresh, indicatore=indicatore)

    def r20(self, freq='1Y', thresh=20, indicatore='R20'):
        scenario_dir = self.scenario.replace(' ', '').replace('.', '')

        # cambia il nome con passato e futuro
        file_indicatore_completo = f'{self.path_indicatori_cache}/{indicatore.lower()}_{self.annoInizio}-{self.annoFine}_{self.annoInizioFuturo}-{self.annoFineFuturo}_{scenario_dir}.nc'
        if os.path.isfile(file_indicatore_completo):
            dataset = xr.open_dataset(file_indicatore_completo)
        else:
            pr = self.getNetcdfModelData('TOT_PREC')

            pr_tr = xr.where(pr > thresh, 0, 1)
            pr_tr = pr_tr.assign_attrs(units='mm/day')

            indicatore_tot = pr_tr.resample(time=freq).sum(dim='time')

            dataset = xr.Dataset({
                indicatore: (['time', 'rlat', 'rlon'], indicatore_tot.values),
                'time': indicatore_tot.time,  # Usa l'intero oggetto time, non solo l'anno
                'rlat': indicatore_tot.rlat,
                'rlon': indicatore_tot.rlon
            })

            # aggiungo le coordinate
            dataset = dataset.assign_coords({'lon': indicatore_tot.lon, 'lat': indicatore_tot.lat})

            dataset.to_netcdf(file_indicatore_completo, mode='w', format="NETCDF4",
                              encoding={'time': {'dtype': 'int32', 'units': 'days since 1900-01-01'}})

        # print(cdd_dataset)
        anomalia = self.crea_anomalia_abs(dataset)

        output_file = f'{self.outputPath}/{self.modelData}_{indicatore.lower()}_{self.annoInizio}-{self.annoFine}_{self.annoInizioFuturo}-{self.annoFineFuturo}_{scenario_dir}.nc'
        anomalia.to_netcdf(output_file, mode='w', format="NETCDF4")
        return anomalia

    def r50(self, freq='1Y', thresh=50, indicatore='R50'):
        self.r20(freq='1Y', thresh=thresh, indicatore=indicatore)

    def sdii(self, freq='YS', indicatore='SDII'):
        """
        SDII - Precipitazione giornaliera (mm/giorno)

        Precipitazione media giornaliera nei giorni di precipitazione maggiore o uguale a 1mm.
        """

        # cambia il nome con passato e futuro
        file_indicatore_completo = f'{self.path_indicatori_cache}/{indicatore.lower()}_1980-2070_{self.scenario_dir}.nc'
        if os.path.isfile(file_indicatore_completo):
            prcptot_dataset = xr.open_dataset(file_indicatore_completo)
        else:
            precip_data = self.getNetcdfModelData(variabile='TOT_PREC')

            # setto a 0 il rumore del modello
            precip_data = xr.where(precip_data < 1, 0, precip_data)

            precip_data = precip_data.assign_attrs(units='mm/day')
            prcptot_result = precip_data.where(precip_data >= 1).resample(time=freq).mean()

            prcptot_dataset = xr.Dataset({
                indicatore: (['time', 'rlat', 'rlon'], prcptot_result.values),
                'time': prcptot_result.time,  # Usa l'intero oggetto time, non solo l'anno
                'rlat': prcptot_result.rlat,
                'rlon': prcptot_result.rlon
            })

            # aggiungo le coordinate
            prcptot_dataset = prcptot_dataset.assign_coords({'lon': prcptot_result.lon, 'lat': prcptot_result.lat})

            # Aggiungi gli attributi
            # cdd_dataset[indicatore].attrs['units'] = 'mm/day'
            '''
            cdd_dataset[indicatore].attrs[
                'long_name'] = 'On a summer day, the maximum temperature is ' + thresh + '. The number of summer days per year is a frequently used climate indicator.'
            cdd_dataset[indicatore].attrs['standard_name'] = 'Summer Days ' + thresh
            cdd_dataset.attrs['reference'] = "https://etccdi.pacificclimate.org/list_27_indices.shtml"
            '''

            prcptot_dataset.to_netcdf(file_indicatore_completo, mode='w', format="NETCDF4",
                                  encoding={'time': {'dtype': 'int32', 'units': 'days since 1900-01-01'}})

        # print(cdd_dataset)
        anomalia = self.crea_anomalia_percentuale(prcptot_dataset)

        output_file = f'{self.outputPath}/{self.modelData}_{indicatore.lower()}_{self.annoInizio}-{self.annoFine}_{self.annoInizioFuturo}-{self.annoFineFuturo}_{self.scenario_dir}.nc'
        anomalia.to_netcdf(output_file, mode='w', format="NETCDF4")
        return anomalia
    
    
    def su(self, thresh=25, freq='1Y', indicatore='SU'):
        """
        Numero di giorni con temperatura massima giornaliera maggiore di 25°C.
        """
        
        file_indicatore_completo = f'{self.path_indicatori_cache}/{indicatore.lower()}_1980-2070_{self.scenario_dir}.nc'

        if os.path.isfile(file_indicatore_completo):
            su_dataset = xr.open_dataset(file_indicatore_completo)
        else:
            tasmax_data = self.getNetcdfModelData('TMAX')
            print(tasmax_data)
            tasmax_data = tasmax_data - 273.15
            tasmax_data.attrs['units'] = 'degC'

            su_result = xr.where(tasmax_data > thresh, 1, 0)
            
            
            

            # orgnizzo i dati: facendo la somma annuale
            su_result = su_result.resample(time = freq).sum(dim='time')

            # Crea un dataset con la variabile SU, mantenendo esplicitamente la dimensione 'time'
            su_dataset = xr.Dataset({
                indicatore: (['time', 'rlat', 'rlon'], su_result.values),
                'time': su_result.time,  # Usa l'intero oggetto time, non solo l'anno
                'rlat': su_result.rlat,
                'rlon': su_result.rlon
            })

            # aggiungo le coordinate
            su_dataset = su_dataset.assign_coords({'lon': su_result.lon, 'lat': su_result.lat})
            '''
            # Aggiungo gli attributi
            su_dataset[indicatore].attrs['units'] = 'degC'
            su_dataset[indicatore].attrs[
                'long_name'] = 'On a summer day, the maximum temperature is ' + thresh + '. The number of summer days per year is a frequently used climate indicator.'
            su_dataset[indicatore].attrs['standard_name'] = 'Summer Days ' + thresh
            su_dataset.attrs['reference'] = "https://etccdi.pacificclimate.org/list_27_indices.shtml"
            '''

            su_dataset.to_netcdf(file_indicatore_completo, mode='w', format="NETCDF4",
                                  encoding={'time': {'dtype': 'int32', 'units': 'days since 1900-01-01'}})


        # print(cdd_dataset)
        anomalia = self.crea_anomalia_abs(su_dataset)

        output_file = f'{self.outputPath}/{self.modelData}_{indicatore.lower()}_{self.annoInizio}-{self.annoFine}_{self.annoInizioFuturo}-{self.annoFineFuturo}_{self.scenario_dir}.nc'
        anomalia.to_netcdf(output_file, mode='w', format="NETCDF4")

        return anomalia

    def su95p(self, freq='1Y'):
        return self.su(thresh=29.2, freq=freq, indicatore='SU95p')

    def tg(self, indicatore='TG', freq='YS'):
        # Paolo
        scenario = self.scenario.replace(' ', '').replace('.', '')
        file_completo = f'{self.path_indicatori_cache}/{indicatore.lower()}_1980-2070_{scenario}.nc'
        if os.path.isfile(file_completo):
            tg = xr.open_dataset(file_completo)
        else:
            ds_tas = self.getNetcdfModelData(variabile = 'T_2M')
            ds_tas = ds_tas - 273.15
            ds_tas.attrs['units'] = 'degC'
            tg_raw = xclim.indices.tg_mean(tas=ds_tas, freq=freq) #la funzione ritorna la temperature media alla data frequenza temporale (1 anno default) 
            tg = xr.Dataset({indicatore: (['time', 'rlat', 'rlon'], tg_raw.values),
                'time': tg_raw.time,  # Usa l'intero oggetto time, non solo l'anno
                'rlat': tg_raw.rlat,
                'rlon': tg_raw.rlon})
            
            tg = tg.assign_coords({'lon' : tg_raw.lon , 'lat' : tg_raw.lat})
            tg.to_netcdf(file_completo, mode='w', format='NETCDF4', encoding={'time': {'dtype': 'int32', 'units': 'days since 1900-01-01'}})

        tg_abs_anomaly = self.crea_anomalia_abs(tg)
        output_file = f'{self.outputPath}/{self.modelData}_{indicatore.lower()}_{self.annoInizio}-{self.annoFine}_{self.annoInizioFuturo}-{self.annoFineFuturo}_{scenario}.nc'
        tg_abs_anomaly.to_netcdf(output_file, mode='w', format="NETCDF4")
        return tg_abs_anomaly    

    def tg95prctile(self, indicatore='TG95prctile', an = 'abs'):
        """
        TG95prctile - 95° percentile della temperatura media (°C)

        95° percentile della temperatura media giornaliera.
        """
        self.tprctile(indicatore=indicatore, variabile='T_2M', percentile=95, an = an)

    def tg99prctile(self, indicatore='TG99prctile', an = 'abs'):
        """
        TG99prctile - 99° percentile della temperatura media (°C)

        99° percentile della temperatura media giornaliera.
        """
        self.tprctile(indicatore=indicatore, variabile='T_2M', percentile=99, an = an)


    def tprctile(self, indicatore='TN__prctile', variabile='TMIN', percentile=95, an = 'abs'):
        """
        calcola il percentile della variabile
        """
        file_indicatore_completo = f'{self.path_indicatori_cache}/{indicatore.lower()}_1980-2070_{self.scenario_dir}.nc'
        
        if os.path.isfile(file_indicatore_completo):
            dataset = xr.open_dataset(file_indicatore_completo)
        else:
            # Calcolo il 95° percentiles con calcolaPercentile
            tmean_95 = self.getNetcdfModelData(variabile = variabile)

            if variabile== 'TOT_PREC':
                # setto a 0 il rumore del modello
                tmean_95 = xr.where(tmean_95 < 1, 0, tmean_95)
                
            bs = tmean_95.sel(time = slice(str(self.annoInizio), str(self.annoFine))).quantile(percentile/100, skipna = True, dim = 'time')
            tl = tmean_95.sel(time = slice(str(self.annoInizioFuturo), str(self.annoFineFuturo))).quantile(percentile/100, skipna = True, dim = 'time')
            
            if an == '%':
                anomalia = ((tl - bs)/ bs) * 100
            else: 
                anomalia = tl - bs
       
        anomalia = anomalia.to_dataset(name = indicatore)

        output_file = f'{self.outputPath}/{self.modelData}_{indicatore.lower()}_{self.annoInizio}-{self.annoFine}_{self.annoInizioFuturo}-{self.annoFineFuturo}_{self.scenario_dir}.nc'
        anomalia.to_netcdf(output_file, mode='w', format="NETCDF4")
        
        return anomalia

    def tn10prctile(self, indicatore='TN10prctile', an = 'abs'):
        """
        TN10prctile- 10° percentile della temperatura minima(°C)

        10° percentile della temperatura minima giornaliera.
        """
        self.tprctile(indicatore=indicatore, variabile='TMIN', percentile=10, an = an)

    def tn5prctile(self, indicatore='TN5prctile', an = 'abs'):
        """
        TN99prctile- 5° percentile della temperatura minima(°C)

        5° percentile della temperatura minima giornaliera.
        """
        self.tprctile(indicatore=indicatore, variabile='TMIN', percentile=5, an = an)

    def tr(self, freq = '1Y', indicatore = 'TR'):


        # cambia il nome con passato e futuro
        file_indicatore_completo = f'{self.path_indicatori_cache}/{indicatore.lower()}_1980-2070_{self.scenario_dir}.nc'
        if os.path.isfile(file_indicatore_completo):
            dataset = xr.open_dataset(file_indicatore_completo)
        else:
            tmin = self.getNetcdfModelData('TMIN_2M')

            # setto a 0 il rumore del modello
            tr = xr.where(tmin > 20 + 273.15, 1, 0)
            
            indicatore_tot = tr.resample(time = freq).sum(dim='time')
            

            dataset = xr.Dataset({
                indicatore: (['time', 'rlat', 'rlon'], indicatore_tot.values),
                'time': indicatore_tot.time,  # Usa l'intero oggetto time, non solo l'anno
                'rlat': indicatore_tot.rlat,
                'rlon': indicatore_tot.rlon
            })

            # aggiungo le coordinate
            dataset = dataset.assign_coords({'lon': indicatore_tot.lon, 'lat': indicatore_tot.lat})


            dataset.to_netcdf(file_indicatore_completo, mode='w', format="NETCDF4",
                                  encoding={'time': {'dtype': 'int32', 'units': 'days since 1900-01-01'}})

        # print(cdd_dataset)
        anomalia = self.crea_anomalia_abs(dataset)

        output_file = f'{self.outputPath}/{self.modelData}_{indicatore.lower()}_{self.annoInizio}-{self.annoFine}_{self.annoInizioFuturo}-{self.annoFineFuturo}_{self.scenario_dir}.nc'
        anomalia.to_netcdf(output_file, mode='w', format="NETCDF4")
        return anomalia

    def tr20pr(self, indicatore='TR20_PR', freq='YS', t=20, mode='max'):
        # Paolo
        scenario = self.scenario.replace(' ', '').replace('.', '')
        output_file = f'{self.outputPath}/{self.modelData}_{indicatore.lower()}_{self.annoInizio}-{self.annoFine}_{self.annoInizioFuturo}-{self.annoFineFuturo}_{scenario}'

        if os.path.isfile(output_file):
            tr_anomaly = xr.open_dataset(output_file)
        else:
            pr = self.getNetcdfModelData('TOT_PREC')

            pr_past = pr.sel(time = slice(str(self.annoInizio) + '-01-01', str(self.annoFine) + '-12-01'))
            tr_past_raw = frequency_analysis(pr_past, t=t, dist='genextreme', mode=mode, freq=freq)
            tr_past = xr.Dataset({
                indicatore: (['return_period', 'rlat', 'rlon'], tr_past_raw.values),
                'lat': (['rlat', 'rlon'], pr.lat.values),
                'lon': (['rlat', 'rlon'], pr.lon.values),
                'return_period': tr_past_raw.return_period, 
                'rlat': tr_past_raw.rlat,
                'rlon': tr_past_raw.rlon})
            del tr_past_raw

            pr_future = pr.sel(time=slice(str(self.annoInizioFuturo) + '-01-01', str(self.annoFineFuturo) + '-12-01'))
            tr_future_raw = frequency_analysis(pr_future, t=t, dist='genextreme', mode=mode, freq=freq)
            tr_future = xr.Dataset({
                indicatore: (['return_period', 'rlat', 'rlon'], tr_future_raw.values),
                'lat': (['rlat', 'rlon'], pr.lat.values),
                'lon': (['rlat', 'rlon'], pr.lon.values),
                'return_period': tr_future_raw.return_period,
                'rlat': tr_future_raw.rlat,
                'rlon': tr_future_raw.rlon})
            del tr_future_raw
 
            tr_anomaly_raw = (tr_future[indicatore] - tr_past[indicatore]) / tr_past[indicatore] * 100
            
            del tr_past
            tr_anomaly = xr.Dataset({
                indicatore: (['return_period', 'rlat', 'rlon'], tr_anomaly_raw.values),
                'lat': (['rlat', 'rlon'], tr_future.lat.values),
                'lon': (['rlat', 'rlon'], tr_future.lon.values),
                'return_period': tr_anomaly_raw.return_period,
                'rlat': tr_anomaly_raw.rlat,
                'rlon': tr_anomaly_raw.rlon})
                        
            tr_anomaly.to_netcdf(output_file, mode='w', format='NETCDF4')
        return 
    
    def tr50pr(self):
        return self.tr20pr(indicatore='TR50_PR', freq = 'YS', t=50, mode='max')

    def tr100pr(self):
        return self.tr20pr(indicatore='TR100_PR', freq = 'YS', t=100, mode='max')    

    
    def tx90prctile(self, indicatore='TX90prctile', an = 'abs'):
        """
        TX90prctile - 90° percentile della temperatura massima (°C)

        90° percentile della temperatura massima giornaliera.
        """
        self.tprctile(indicatore=indicatore, variabile='TMAX', percentile=90, an = an) 

    def tx95prctile(self, indicatore='TX95prctile', an = 'abs'):
        """
        TX95prctile - 95° percentile della temperatura massima (°C)

        95° percentile della temperatura massima giornaliera.
        """
        self.tprctile(indicatore=indicatore, variabile='TMAX', percentile=95, an = an)

    def tx99prctile(self, indicatore='TX99prctile', an = 'abs'):
        """
        TX99prctile - 99° percentile della temperatura massima (°C)

        99° percentile della temperatura massima giornaliera.
        """
        self.tprctile(indicatore=indicatore, variabile='TMAX', percentile=99, an = an)


    def wsdi(self, indicatore = 'WSDI', window = 6, freq = 'YS', op = '>'):
        
        scenario_dir = self.scenario.replace(' ', '').replace('.', '')
        tasmax_per_ds = self.calcolaPercentile('TMAX_2M', window=5, percentile=90)
        
        tasmax_per = tasmax_per_ds - 273.15
        tasmax_per.attrs['units'] = 'degC'
        
        file_indicatore_completo = f'{self.path_indicatori_cache}/{indicatore.lower()}_{self.annoInizio}-{self.annoFine}_{self.annoInizioFuturo}-{self.annoFineFuturo}_{scenario_dir}.nc'
        
        
        if os.path.isfile(file_indicatore_completo):
            dataset = xr.open_dataset(file_indicatore_completo)
        else:
            tasmax = self.getNetcdfModelData('TMAX_2M')

        
        
            indicatore_tot = xclim.indices.warm_spell_duration_index(tasmax, tasmax_per, window=window, freq=freq, op=op)
            
            dataset = xr.Dataset({
                indicatore: (['time', 'rlat', 'rlon'], indicatore_tot.values),
                'time': indicatore_tot.time,  # Usa l'intero oggetto time, non solo l'anno
                'rlat': indicatore_tot.rlat,
                'rlon': indicatore_tot.rlon
            })

            # aggiungo le coordinate
            dataset = dataset.assign_coords({'lon': indicatore_tot.lon, 'lat': indicatore_tot.lat})


            dataset.to_netcdf(file_indicatore_completo, mode='w', format="NETCDF4",
                                  encoding={'time': {'dtype': 'int32', 'units': 'days since 1900-01-01'}})

        # print(cdd_dataset)
        anomalia = self.crea_anomalia_abs(dataset)

        output_file = f'{self.outputPath}/{self.modelData}_{indicatore.lower()}_{self.annoInizio}-{self.annoFine}_{self.annoInizioFuturo}-{self.annoFineFuturo}_{scenario_dir}.nc'
        anomalia.to_netcdf(output_file, mode='w', format="NETCDF4")
        return anomalia

    # alebo
    def wd(self, indicatore='WD', freq='YS'):
        """
        WD-Gioni caldi-secchi (giorni)
        
        Numero di giorni con temperatura media giornaliera maggiore del 75° percentile** della temperatura media giornaliera 
        e con precipitazione giornaliera minore del 75° percentile** della precipitazione giornaliera.
        """
        file_indicatore_completo = f'{self.path_indicatori_cache}/{indicatore.lower()}_1980-2070_{self.scenario_dir}.nc'
        
        if os.path.isfile(file_indicatore_completo):
            dataset = xr.open_dataset(file_indicatore_completo)
        else:
            # Temperatura media giornaliera
            tmean = self.getNetcdfModelData('T_2M')

            # Precipitazione giornaliera
            prec = self.getNetcdfModelData('TOT_PREC')
            
            # Calcolo il 75° percentiles con calcolaPercentile
            tmean_75p = self.calcolaPercentile('T_2M', window=5, percentile=75)
            prec_75p = self.calcolaPercentile('TOT_PREC', window=5, percentile=75)

            # Trovo i Gioni caldi-secchi con xclim
            wd_count = xclim.indices.warm_and_dry_days(tmean, prec, tmean_75p, prec_75p, freq=freq)
            
            # creo il dataset espicitando tempo e dimensioni
            dataset = xr.Dataset({
                indicatore: (['time', 'rlat', 'rlon'], wd_count.values),
                'time': wd_count.time,
                'rlat': wd_count.rlat,
                'rlon': wd_count.rlon
            })
            
            # aggiungo le coordinate
            dataset = dataset.assign_coords({'lon': wd_count.lon, 'lat': wd_count.lat})
            
            # salvo il file per l'intero periodo
            dataset.to_netcdf(file_indicatore_completo, mode='w', format="NETCDF4",
                            encoding={'time': {'dtype': 'int32', 'units': 'days since 1900-01-01'}})
        
        # Calcolo anomalia assoluta
        anomalia = self.crea_anomalia_abs(dataset)
        
        # Salvo anomalia
        output_file = f'{self.outputPath}/{self.modelData}_{indicatore.lower()}_{self.annoInizio}-{self.annoFine}_{self.annoInizioFuturo}-{self.annoFineFuturo}_{self.scenario_dir}.nc'
        anomalia.to_netcdf(output_file, mode='w', format="NETCDF4")
        
        return anomalia
    
    # alebo
    def ww(self, indicatore='WW', freq='YS'):
        """
        WW-Giorni caldi-piovosi (giorni) 
        
        Numero di giorni con temperatura media giornaliera maggiore del 75° percentile** della temperatura media giornaliera 
        e con precipitazione giornaliera maggiore del 75° percentile** della precipitazione giornaliera.
        """
        file_indicatore_completo = f'{self.path_indicatori_cache}/{indicatore.lower()}_1980-2070_{self.scenario_dir}.nc'
        
        if os.path.isfile(file_indicatore_completo):
            dataset = xr.open_dataset(file_indicatore_completo)
        else:
            # Temperatura media giornaliera
            tmean = self.getNetcdfModelData('T_2M')

            # Precipitazione giornaliera
            prec = self.getNetcdfModelData('TOT_PREC')
            
            # Calcolo il 75° percentiles con calcolaPercentile
            tmean_75p = self.calcolaPercentile('T_2M', window=5, percentile=75)
            prec_75p = self.calcolaPercentile('TOT_PREC', window=5, percentile=75)
            
            # Conto i Giorni caldi-piovosi per [freq]
            ww_count = xclim.indices.warm_and_wet_days(tmean, prec, tmean_75p, prec_75p, freq=freq)
            
            # creo il dataset espicitando tempo e dimensioni
            dataset = xr.Dataset({
                indicatore: (['time', 'rlat', 'rlon'], ww_count.values),
                'time': ww_count.time,
                'rlat': ww_count.rlat,
                'rlon': ww_count.rlon
            })
            
            # aggiungo le coordinate
            dataset = dataset.assign_coords({'lon': ww_count.lon, 'lat': ww_count.lat})
            
            # salvo il file per l'intero periodo
            dataset.to_netcdf(file_indicatore_completo, mode='w', format="NETCDF4",
                            encoding={'time': {'dtype': 'int32', 'units': 'days since 1900-01-01'}})
        
        # Calcolo anomalia assoluta
        anomalia = self.crea_anomalia_abs(dataset)
        
        # Salvo anomalia
        output_file = f'{self.outputPath}/{self.modelData}_{indicatore.lower()}_{self.annoInizio}-{self.annoFine}_{self.annoInizioFuturo}-{self.annoFineFuturo}_{self.scenario_dir}.nc'
        anomalia.to_netcdf(output_file, mode='w', format="NETCDF4")
        
        return anomalia
    
    def pr95prctile(self, indicatore='PR90PRCTILE', an = '%'):
        
        self.tprctile(indicatore=indicatore, variabile='TOT_PREC', percentile=90, an = an)
        
    def pr95prctile(self, indicatore='PR95PRCTILE', an = '%'):
        
        self.tprctile(indicatore=indicatore, variabile='TOT_PREC', percentile=95, an = an)

    def pr99prctile(self, indicatore='PR99PRCTILE', an = '%'):
        
        self.tprctile(indicatore=indicatore, variabile='TOT_PREC', percentile=99, an = an)
    
    def drought_spi(self, tp_daily, baseline = (1981, 2011), scale=3):
        from climate_indices import indices as cindx
        """ Using Standard Precipitation index """

        ## total monthly precipitation
        #tp_monthly = tp_daily.resample(time='MS').sum(dim='time')
        tp_monthly = tp_daily.resample(time='MS').sum(dim = 'time')  # Riga corretta
        
        ## Wrap daily precipitation function
        DISTRIBUTION = cindx.Distribution['gamma']
        DATA_START_YEAR = int(tp_daily.isel(time=0)['time.year'])
        CALIBRATION_YEAR_INITIAL = baseline[0]
        CALIBRATION_YEAR_FINAL = baseline[1] - 1
        PERIODICITY = cindx.compute.Periodicity['monthly']
        compute_spi =   lambda x : cindx.spi(x,
                                            scale,
                                            DISTRIBUTION,
                                            DATA_START_YEAR,
                                            CALIBRATION_YEAR_INITIAL,
                                            CALIBRATION_YEAR_FINAL,
                                            PERIODICITY)

        ## compute SPI/gamma at 3-month scale
        tp_monthly = tp_monthly.chunk({'time' : -1}) # re-chunk along time axis
        time_dim = tp_monthly.sizes['time']

        stand_prec_index = xr.apply_ufunc(compute_spi,
                                tp_monthly,
                                input_core_dims=[['time']],
                                exclude_dims=set(('time',)),
                                output_core_dims=[['time']],
                                output_sizes={'time': time_dim},
                                dask = 'parallelized',
                                output_dtypes  = [float],
                                vectorize = True
                                )

        ## setting up the original time axis
        old_time_ax = tp_monthly['time'].values
        stand_prec_index = stand_prec_index.assign_coords(time=old_time_ax)
        #print(list(stand_prec_index.variables.keys())[0] )
        prec_label = list(stand_prec_index.variables.keys())[0]
        #print(stand_prec_index)
        stand_prec_index = stand_prec_index.rename({ prec_label : 'drought'})
        stand_prec_index['drought'].attrs['long_name'] = 'drought index '
        stand_prec_index['time'].attrs['long_name'] = 'time'

        return  stand_prec_index
    
    
    def drought_spei(self, daily_water_budget, baseline = (1981, 2011), scale=3):
        from climate_indices import indices as cindx
        """ Using Standard Precipitation index """

        ## Wrap daily precipitation function
        DISTRIBUTION = cindx.Distribution['pearson']
        DATA_START_YEAR = int(daily_water_budget.isel(time=0)['time.year'])
        CALIBRATION_YEAR_INITIAL = baseline[0]
        CALIBRATION_YEAR_FINAL = baseline[1] - 1
        PERIODICITY = cindx.compute.Periodicity['monthly']
        compute_spei =   lambda x : cindx.spi(x,
                                            scale,
                                            DISTRIBUTION,
                                            DATA_START_YEAR,
                                            CALIBRATION_YEAR_INITIAL,
                                            CALIBRATION_YEAR_FINAL,
                                            PERIODICITY)
        print(compute_spei)

        ## compute SPEI/pearson at scale-month scale
        tp_monthly = daily_water_budget.chunk({'time' : -1}) # re-chunk along time axis
        time_dim = tp_monthly.sizes['time']

        stand_prec_index = xr.apply_ufunc(compute_spei,
                                tp_monthly,
                                input_core_dims=[['time']],
                                exclude_dims=set(('time',)),
                                output_core_dims=[['time']],
                                output_sizes={'time': time_dim},
                                dask = 'parallelized',
                                output_dtypes  = [float],
                                vectorize = True
                                )

        ## setting up the original time axis
        old_time_ax = tp_monthly['time'].values
        stand_prec_index = stand_prec_index.assign_coords(time=old_time_ax)
        #print(list(stand_prec_index.variables.keys())[0] )
        prec_label = list(stand_prec_index.variables.keys())[0]
        #print(stand_prec_index)
        stand_prec_index = stand_prec_index.rename({ prec_label : 'drought'})
        stand_prec_index['drought'].attrs['long_name'] = 'drought index '
        stand_prec_index['time'].attrs['long_name'] = 'time'

        return  stand_prec_index
    
    
    def spi3(self, indicatore = 'SPI3', scale=3):
        file_indicatore_completo = f'{self.outputPath}/{self.modelData}_{indicatore.lower()}_{self.annoInizio}-{self.annoFine}_{self.annoInizioFuturo}-{self.annoFineFuturo}_{self.scenario_dir}.nc'

        cal_start = '1981-01-01'
        cal_end='2010-12-01'

        file_path = f'{self.path_variabili}/TOT_PREC_VHR-PRO-IT_1981_2070_{self.scenario_dir}.nc'
        tp_daily = xr.open_dataset(file_path)
        #tp_daily=tp_daily['pr']

        spi = self.drought_spi(tp_daily, scale=scale)
        

        classes = [(-2, float('-inf'), 'Extremely_dry'),
                    (-1.5, -2, 'Severely_dry'),
                    (-1, -1.5, 'Moderately_dry'),
                    (1, -1, 'Near_normal'),
                    (1.5, 1, 'Moderately_wet'),
                    (2, 1.5, 'Very_wet'),
                    (float('inf'), 2, 'Extremely_wet')]

        
        ds = xr.Dataset()
        for thresholds in classes:
            spi_one_class = spi['drought'].where((spi['drought'] < thresholds[0]) & (spi['drought'] > thresholds[1]))

            spi_benchmark = spi_one_class.sel(time = slice(cal_start, cal_end))
            spi_scenario = spi_one_class.sel(time = slice(str(self.annoInizioFuturo) + '-01-01', str(self.annoFineFuturo) + '-12-01'))

            percentage_future = spi_scenario.count(dim='time') / len(spi_scenario.time) * 100
            percentage_past = spi_benchmark.count(dim='time') / len(spi_benchmark.time) * 100

            spi_anomaly = percentage_future - percentage_past

            ds[thresholds[2]] = spi_anomaly
            
        ds = ds.assign_coords({'lon': tp_daily.lon, 'lat': tp_daily.lat, 'rlon': tp_daily.rlon, 'rlat': tp_daily.rlat})

        ds.to_netcdf(file_indicatore_completo, mode='w', format="NETCDF4")
    
    def spei3(self, indicatore = 'SPEI3', scale=3):
        file_indicatore_completo = f'{self.outputPath}/{self.modelData}_{indicatore.lower()}_{self.annoInizio}-{self.annoFine}_{self.annoInizioFuturo}-{self.annoFineFuturo}_{self.scenario_dir}.nc'

        cal_start = '1981-01-01'
        cal_end='2010-12-01'

        file_path = f'{self.outputPath}/VHR-PRO-IT_dwb_1981-2010_2036-2065_{self.scenario_dir}.nc'
        tp_daily = xr.open_dataset(file_path)
        #tp_daily=tp_daily['pr']

        spi = self.drought_spei(tp_daily, scale=scale)
        

        classes = [(-2, float('-inf'), 'Extremely_dry'),
                    (-1.5, -2, 'Severely_dry'),
                    (-1, -1.5, 'Moderately_dry'),
                    (1, -1, 'Near_normal'),
                    (1.5, 1, 'Moderately_wet'),
                    (2, 1.5, 'Very_wet'),
                    (float('inf'), 2, 'Extremely_wet')]

        
        ds = xr.Dataset()
        for thresholds in classes:
            spi_one_class = spi['drought'].where((spi['drought'] < thresholds[0]) & (spi['drought'] > thresholds[1]))

            spi_benchmark = spi_one_class.sel(time = slice(cal_start, cal_end))
            spi_scenario = spi_one_class.sel(time = slice(str(self.annoInizioFuturo) + '-01-01', str(self.annoFineFuturo) + '-12-01'))

            percentage_future = spi_scenario.count(dim='time') / len(spi_scenario.time) * 100
            percentage_past = spi_benchmark.count(dim='time') / len(spi_benchmark.time) * 100

            spi_anomaly = percentage_future - percentage_past

            ds[thresholds[2]] = spi_anomaly
            
        ds = ds.assign_coords({'lon': tp_daily.lon, 'lat': tp_daily.lat, 'rlon': tp_daily.rlon, 'rlat': tp_daily.rlat})

        ds.to_netcdf(file_indicatore_completo, mode='w', format="NETCDF4")


    def daily_water_budget(self, indicatore = 'DWB'):
        
        file_indicatore_completo = f'{self.outputPath}/{self.modelData}_{indicatore.lower()}_{self.annoInizio}-{self.annoFine}_{self.annoInizioFuturo}-{self.annoFineFuturo}_{self.scenario_dir}.nc'

        t2m = self.getNetcdfModelData('T_2M')
        
        tp = self.getNetcdfModelData('TOT_PREC')
        
        month_tp = tp.resample(time = '1M').sum(dim = 'time')
        del tp
        
        pet = xclim.indices.potential_evapotranspiration(method='TW48', tas = t2m) * 2630000
        
        del t2m
        
        dwb = month_tp - pet.values
        
        dwb.to_dataset(name = indicatore).to_netcdf(file_indicatore_completo)
        
    
        
