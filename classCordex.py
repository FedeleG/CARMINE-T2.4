import os
import sys
import xarray as xr
import xclim
from xclim.core.calendar import percentile_doy
from xclim.indices.stats import frequency_analysis
from xclim import units
#current_dir = os.path.dirname(os.path.abspath(__file__))
#parent_dir = os.path.dirname(current_dir)
#sys.path.append(parent_dir)
import re
import numpy as np
from classeIndicatori import Indicatori


class Cordex(Indicatori):
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

    def getNetcdfModelData(self, variabile='TMAX'):
        units = ''
        standard_name = ''
        long_name = ''
    
        if variabile in ('TMAX', 'TMAX_2M', 'tasmax'):
            variabile = "TMAX_2M"
            units = 'degK'
            standard_name = 'air_temperature'
            long_name = 'Daily maximum near-surface air temperature'
            if self.modelData == "EU-CORDEX-11" or self.modelData =='CMIP6': variabile = 'tasmax'
        elif variabile in ('T_2M', 'tas'):
            variabile = "T_2M"
            units = 'degK'
            standard_name = 'air_temperature'
            long_name = 'Daily minimum near-surface air temperature'
            if self.modelData == "EU-CORDEX-11" or self.modelData =='CMIP6': variabile = 'tas'
        elif variabile in ('TMIN', 'TMIN_2M', 'tasmin'):
            variabile = "TMIN_2M"
            units = 'degK'
            standard_name = 'air_temperature'
            long_name = 'Daily mean near-surface air temperature'
            if self.modelData == "EU-CORDEX-11" or self.modelData =='CMIP6': variabile = 'tasmin'
        elif variabile in ('TOT_PREC', 'pr'):
            variabile = "pr"
            units = 'mm/day'
            standard_name = 'precipitation amount'
            long_name = 'total precipitation amount'
            if self.modelData == "EU-CORDEX-11" or self.modelData =='CMIP6': variabile = 'pr'
        elif variabile in ('WIND', 'sfcWind'):
            variabile = "WIND"
            units = 'm/s'
            standard_name = 'maximum daily wind speed'
            long_name = 'maximum daily wind speed'
            if self.modelData == "EU-CORDEX-11" or self.modelData =='CMIP6': variabile = 'sfcWind'
        elif variabile in ('HURS', 'hurs'):
            variabile = "HURS"
            units = '%'
            standard_name = 'relative_humidity'
            long_name = 'Near-Surface Relative Humidity'
            if self.modelData == "EU-CORDEX-11" or self.modelData =='CMIP6': variabile = 'hurs'
        else:
            print('Variabile non trovata ' + variabile)
            return None
    
        scenario = self.scenario.replace(' ', '').replace('.', '')
    
        if self.modelData == "CMIP6":
            lista_dataset = []
            for modellist in range(1, 33):
                modello = f'{self.path_variabili}/{variabile}_cmip6_model{modellist:02d}_{scenario.lower()}.nc'
                if os.path.isfile(modello):
                    lista_dataset.append(modello)
            return lista_dataset
    
        if self.modelData == "EU-CORDEX-11":
            lista_dataset = []
            for modellist in range(1, 33):
                modello = f'{self.path_variabili}/{variabile}_cordex_model{modellist:02d}_{scenario.lower()}.nc'
                if os.path.isfile(modello):
                    lista_dataset.append(modello)
            return lista_dataset
    
        if self.modelData == "VHR-PRO-IT":
            print(f'{self.path_variabili}/{variabile}_VHR-PRO-IT_1981_2070_{scenario}.nc')
            dsdata = xr.open_dataset(f'{self.path_variabili}/{variabile}_VHR-PRO-IT_1981_2070_{scenario}.nc')
    
            dsdata = dsdata[variabile]
            data = self.resample(dsdata, variabile)
            data.attrs["units"] = units
            data.attrs['standard_name'] = standard_name
            data.attrs['long_name'] = long_name
    
            return data
        else:
            print(f"Modello non riconosciuto: {self.modelData}")
            return None

            
    def getNetcdfOpenDataset(self, modello, variabile):
        dsdata = xr.open_dataset(modello)
        
        # Estraggo la variabile
        dsdata = dsdata[variabile]
        if self.modelData == "EU-CORDEX-11" and variabile=='pr':
            '''
            La pr dei modelli va moltiplicata per un fattore correttivo pari a 86400 perchè l'unità di misura cordex 
            è un flusso di precipitazione che ha come unità di misura kg m-2 s-1
            '''
            dsdata = dsdata*86400
        # Faccio il resample per ogni giorno
        data = self.resample(dsdata, variabile)
        return data
  
    def calcolaPercentile(self, ds, percentile=99, dim='time'):
        """
        Calcolo del percentile
        ds: dataset della variabile
        percentile: il percentile da calcolare
        dim: dimensione 
        """
        percentuale = percentile/100
        valore_percentile=ds.quantile(percentuale, dim=dim, skipna=True)
        return valore_percentile
    
    def _cdd(self, thresh='1 mm/day', freq='1YE', indicatore='HW', model=None):
        """
        Calcolo dell'indicatore Consecutive Dry Days (CDD)
        
        Parameters:
        -----------
        thresh : float, default=35
            Soglia di temperatura in °C per definire un giorno di ondata di calore
        freq : str, default='1Y'
            Frequenza di aggregazione temporale
        indicatore: str, optional
            nome  dell'indicatore    
        model : str, optional
            Percorso del file del modello specifico
        """
        # Gestione del modello e del percorso di output
        if model is not None:
            # Estrai il numero del modello dall'path
            regex = r"model([0-9]+)_rcp"
            matches = re.finditer(regex, model, re.DOTALL)
            modelloNumero = [match.group(1) for match in matches][0]
            
            # Carica i dati e rinomina le coordinate
            pr_data = self.getNetcdfOpenDataset(model, variabile='pr')
            try:
                pr_data = pr_data.rename({'y': 'rlat', 'x': 'rlon'})
            except: 
                print('no')
            file_indicatore_completo = f'{self.path_indicatori_cache}/{self.modelData}_{indicatore.lower()}_MODEL{modelloNumero}_{self.scenario_dir}.nc'
        else:
            pr_data = self.getNetcdfModelData(variabile='TOT_PREC')
            file_indicatore_completo = f'{self.path_indicatori_cache}/{indicatore.lower()}_1980-2070_{self.scenario_dir}.nc'
    
        # setto a 0 il rumore del modello
        pr = xr.where(pr_data < 1, 0, pr_data)

        pr = pr.assign_attrs(units='mm/day')
        pr = pr.fillna(np.nan)
        try:
            indicatore_tot = xclim.indices.maximum_consecutive_dry_days(pr, thresh=thresh, freq=freq,
                                                              resample_before_rl=True)
        except Exception as e:
            print(f"Errore nel processare xclim.indices.maximum_consecutive_dry_days: {e}")

            
        cdd_dataset = xr.Dataset({
            indicatore: (['time', 'rlat', 'rlon'], indicatore_tot.values),
            'time': indicatore_tot.time,  # Usa l'intero oggetto time, non solo l'anno
            'rlat': indicatore_tot.rlat,
            'rlon': indicatore_tot.rlon
        })

        # aggiungo le coordinate
        cdd_dataset = cdd_dataset.assign_coords({'lon': indicatore_tot.lon, 'lat': indicatore_tot.lat})


        cdd_dataset.to_netcdf(file_indicatore_completo, mode='w', format="NETCDF4",
                                  encoding={'time': {'dtype': 'int32', 'units': 'days since 1900-01-01'}})
        
        return cdd_dataset

    def cdd(self, indicatore='CDD', thresh='1 mm/day', freq='YS'):
        model_list = self.getNetcdfModelData(variabile='pr')
        lista_completa = []
        
        # Processa ogni modello
        for modello in model_list:
            print(f"Processando modello: {modello}")
            
            # calcolo l'indicatore per ogni modello
            miomodello = self._cdd(freq=freq, thresh=thresh, indicatore=indicatore, model=modello)

            # calcolo anomalia del modello
            anomalia = self.crea_anomalia_abs(miomodello)

            # aggiungo l'anomalia alla lista
            lista_completa.append(anomalia)
            

        # Calcola la media dell'ensemble
        tutti_modelli = xr.concat(lista_completa, dim='time')
        tutti_modelli_mean = tutti_modelli.mean(dim='time')
        tutti_modelli_std = tutti_modelli.std(dim='time')
        tutti_modelli_mean[f"{indicatore}_STD"] = tutti_modelli_std[indicatore]
        tutti_modelli_mean = tutti_modelli_mean.assign_coords({'lon': lista_completa[0].lon, 'lat': lista_completa[0].lat})

        # Salva il risultato finale
        output_file = f'{self.outputPath}/{self.modelData}_{indicatore.lower()}_{self.annoInizio}-{self.annoFine}_{self.annoInizioFuturo}-{self.annoFineFuturo}_{self.scenario_dir}.nc'
        tutti_modelli_mean.to_netcdf(output_file, mode='w', format="NETCDF4")
        print("File salvato:", output_file)
        return model_list
    
    def fd(self, thresh=0, freq='1YE', indicatore='FD'):
        '''
        Numero di giorni con temperatura minima giornaliera minore di 20°
        '''
        self.hw(thresh=thresh, freq=freq, indicatore=indicatore, variabile='tasmin', operatore='<')

    def humidex(self, indicatore='HUMIDEX5', freq='1YE', classe=0, model=None):
        
        """
        calcola l'indicatore tramite xclim

        humidex 1 index < 20
        humidex 2 index >= 20 < 30
        humidex 3 >=30 < 40
        humidex 4 >=40 < 45
        humidex 5 index > 45

        """
        lista_completa = []
        if model is not None:
            # Estrai il numero del modello dall'path
            regex = r"model([0-9]+)_rcp"
            matches = re.finditer(regex, model, re.DOTALL)
            modelloNumero = [match.group(1) for match in matches][0]
            modellotasmax = model.replace('hurs', 'tasmax')
            # Carica i dati e rinomina le coordinate
            # Temperatura mmassima giornaliera
            print(modellotasmax)
            tasmax = self.getNetcdfOpenDataset(modellotasmax, variabile='tasmax') 
            try:
                tasmax = tasmax.rename({'y': 'rlat', 'x': 'rlon'})
            except:
                print('no')
            # Precipitazione giornaliera
            print(model)
            rh = self.getNetcdfOpenDataset(model, variabile='hurs')
            try:
                rh = rh.rename({'y': 'rlat', 'x': 'rlon'})
            except:
                print('no')
            #  Humidex
            humidex= xclim.indices.humidex(tasmax, hurs=rh)
            print(tasmax)
            print(rh)
            print(humidex)

            # trasformo in gradi da celsius
            humidex=humidex-273.15
            
            if classe == 1:
                indicatore_tot = xr.where(humidex < 20, 1, 0)
            elif classe == 2:
                indicatore_tot = xr.where((humidex >= 20) & (humidex < 30), 1, 0)
            elif classe == 3:
                indicatore_tot = xr.where((humidex >= 30) & (humidex < 40), 1, 0)
            elif classe == 4:
                indicatore_tot = xr.where((humidex >= 40) & (humidex < 45), 1, 0)
            elif classe == 5:
                indicatore_tot = xr.where(humidex >= 45, 1, 0)


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

        return dataset

    def humidex1 (self,indicatore='HUMIDEX1', freq='1YE',classe=1):

        model_list = self.getNetcdfModelData(variabile='hurs')
        lista_completa = []
        
        # Processa ogni modello
        for modello in model_list:
            print(f"Processando modello: {modello}")
            
            # calcolo l'indicatore per ogni modello
            miomodello = self.humidex(indicatore=indicatore, freq=freq, classe=classe, model=modello)

            # calcolo anomalia assolute del modello
            anomalia = self.crea_anomalia_abs(miomodello)

            # aggiungo l'anomalia alla lista
            lista_completa.append(anomalia)

        # Calcola la media dell'ensemble
        tutti_modelli = xr.concat(lista_completa, dim='time')
        #ensamble mean
        tutti_modelli_mean = tutti_modelli.mean(dim='time')
        tutti_modelli_std = tutti_modelli.std(dim='time')
        tutti_modelli_mean[f"{indicatore}_STD"] = tutti_modelli_std[indicatore]
        
        tutti_modelli_mean = tutti_modelli_mean.assign_coords({'lon': lista_completa[0].lon, 'lat': lista_completa[0].lat})
        # Salva il risultato finale
        output_file = f'{self.outputPath}/{self.modelData}_{indicatore.lower()}_{self.annoInizio}-{self.annoFine}_{self.annoInizioFuturo}-{self.annoFineFuturo}_{self.scenario_dir}.nc'
        tutti_modelli_mean.to_netcdf(output_file, mode='w', format="NETCDF4")
        print("File salvato:", output_file)
        return tutti_modelli

    def humidex2 (self):
        self.humidex1(indicatore='HUMIDEX2', freq='1YE', classe=2)

    def humidex3 (self):
        self.humidex1(indicatore='HUMIDEX3', freq='1YE', classe=3)

    def humidex4 (self):
        self.humidex1(indicatore='HUMIDEX4', freq='1YE', classe=4)

    def humidex5 (self):
        self.humidex1(indicatore='HUMIDEX5', freq='1YE', classe=5)

    
    def _hw(self, thresh=35, freq='1Y', indicatore='HW', model=None, variabile='TMAX_2M', operatore='>'):
        """
        Calcolo dell'indicatore Heat Waves (HW)
        
        Parameters:
        -----------
        thresh : float, default=35
            Soglia di temperatura in °C per definire un giorno di ondata di calore
        freq : str, default='1Y'
            Frequenza di aggregazione temporale
        model : str, optional
            Percorso del file del modello specifico
        """
        # Gestione del modello e del percorso di output
        if model is not None:
            # Estrai il numero del modello dall'path
            regex = r"model([0-9]+)_rcp"
            matches = re.finditer(regex, model, re.DOTALL)
            modelloNumero = [match.group(1) for match in matches][0]
            
            # Carica i dati e rinomina le coordinate
            tasmax_data = self.getNetcdfOpenDataset(model, variabile)
            try:
                tasmax_data = tasmax_data.rename({'y': 'rlat', 'x': 'rlon'})
            except:
                print('no')
            file_indicatore_completo = f'{self.path_indicatori_cache}/{self.modelData}_{indicatore.lower()}_MODEL{modelloNumero}_{self.scenario_dir}.nc'
        else:
            tasmax_data = self.getNetcdfModelData(variabile)
            file_indicatore_completo = f'{self.path_indicatori_cache}/{indicatore.lower()}_1980-2070_{self.scenario_dir}.nc'
    
        # Log dei valori prima della conversione
        print(f'Controllo i valori min/max prima della conversione in °C')
        print(f'PRIMA Min:{tasmax_data.values.min()} Max:{tasmax_data.values.max()}')
    
        # Conversione da K a °C
        tasmax_data = tasmax_data - 273.15
        tasmax_data.attrs['units'] = 'degC'
        
        # print(f'DOPO Min:{tasmax_data.values.min()} Max:{tasmax_data.values.max()}')
    
        if operatore=='>':
            # Calcolo dei giorni sopra soglia
            hw_days = xr.where(tasmax_data > thresh, 1, 0)
        
        if operatore=='<':
            # Calcolo dei giorni sotto soglia
            hw_days = xr.where(tasmax_data < thresh, 1, 0)

        # Aggregazione annuale
        hw_annual = hw_days.resample(time=freq).sum(dim='time')
        
        # Creazione del dataset
        hw_dataset = xr.Dataset({
            indicatore: (['time', 'rlat', 'rlon'], hw_annual.values),
            'time': hw_annual.time,
            'rlat': hw_annual.rlat,
            'rlon': hw_annual.rlon
        })
        
        # Aggiunta delle coordinate geografiche
        hw_dataset = hw_dataset.assign_coords({
            'lon': hw_annual.lon, 
            'lat': hw_annual.lat
        })
        
        # Aggiunta dei metadati
        hw_dataset[indicatore].attrs.update({
            'units': 'days',
            'long_name': f'Number of days with maximum temperature above {thresh}°C',
            'threshold': f'{thresh}°C'
        })
        
        # Calcolo dell'anomalia usando la funzione esistente
        anomalia = self.crea_anomalia_abs(hw_dataset)
        #print(f'PRIMA Min:{np.min(anomalia)} Max:{np.max(anomalia)}')
        # Salvataggio del risultato
        anomalia.to_netcdf(file_indicatore_completo, mode='w', format="NETCDF4")
        
        return file_indicatore_completo

    def hw(self, thresh=35, freq='1YE', indicatore='HW', variabile='tasmax', operatore='>'):
        """
        Calcolo dell'ensemble mean dell'indicatore HW per tutti i modelli
        """
        # Ottieni la lista dei modelli
        model_list = self.getNetcdfModelData(variabile)
        lista_completa = []
        
        # Processa ogni modello
        for modello in model_list:
            print(f"Processando modello: {modello}")
            try:
                miomodello = self._hw(freq=freq, thresh=thresh, indicatore=indicatore, model=modello, operatore=operatore, variabile=variabile)
                lista_completa.append(miomodello)
            except Exception as e:
                print(f"Errore nel processare il modello {modello}: {e}")
                continue
        
        # Carica tutti i dataset processati
        datasets = [xr.open_dataset(file) for file in lista_completa]
        
        # Aggiungi dimensione temporale per l'ensemble
        datasets_with_time = [ds.expand_dims(time=[i]) for i, ds in enumerate(datasets)]
        
        # Calcola la media dell'ensemble
        tutti_modelli = xr.concat(datasets_with_time, dim='time')
        tutti_modelli_mean = tutti_modelli.mean(dim='time')
        tutti_modelli_std = tutti_modelli.std(dim='time')
        tutti_modelli_mean[f"{indicatore}_STD"] = tutti_modelli_std[indicatore]

        # Aggiunta delle coordinate geografiche
        tutti_modelli_mean = tutti_modelli_mean.assign_coords({
            'lon': datasets[0].lon, 
            'lat': datasets[0].lat
        })
        
        # Salva il risultato finale
        output_file = f'{self.outputPath}/{self.modelData}_{indicatore.lower()}_{self.annoInizio}-{self.annoFine}_{self.annoInizioFuturo}-{self.annoFineFuturo}_{self.scenario_dir}.nc'
        tutti_modelli_mean.to_netcdf(output_file, mode='w', format="NETCDF4")
        print("File salvato:", output_file)
        return model_list
        
    def hw37(self, thresh=37, freq='1YE', indicatore='HW37'):
        self.hw(thresh=thresh, freq=freq, indicatore=indicatore)
    
    def sdii(self, freq='YS', indicatore='SDII'):
        """
        SDII - Precipitazione giornaliera (mm/giorno)

        Precipitazione media giornaliera nei giorni di precipitazione maggiore o uguale a 1mm.
        """
        model_list = self.getNetcdfModelData('pr')
        lista_completa = []
        
        # Processa ogni modello
        for modello in model_list:
            print(f"Processando modello: {modello}")
        
            sdii_data = self.getNetcdfOpenDataset(modello, variabile='pr')
            try:
                sdii_data = sdii_data.rename({'y': 'rlat', 'x': 'rlon'})
            except:
                print('no')
            # setto a 0 il rumore del modello
            sdii_data = xr.where(sdii_data < 1, 0, sdii_data)

            sdii_data = sdii_data.assign_attrs(units='mm/day')
            sdii_result = sdii_data.where(sdii_data >= 1).resample(time=freq).mean()
            
            sdii_dataset = xr.Dataset({
                indicatore: (['time', 'rlat', 'rlon'], sdii_result.values),
                'time': sdii_result.time,  # Usa l'intero oggetto time, non solo l'anno
                'rlat': sdii_result.rlat,
                'rlon': sdii_result.rlon
            })

            # aggiungo le coordinate
            sdii_dataset = sdii_dataset.assign_coords({'lon': sdii_result.lon, 'lat': sdii_result.lat})
            
            anomalia = self.crea_anomalia_percentuale(sdii_dataset)
            
            lista_completa.append(anomalia)
        
        datasets_with_time = [ds.expand_dims(time=[i]) for i, ds in enumerate(lista_completa)]
        tutti_modelli = xr.concat(datasets_with_time, dim='time')

        # fare la media
        tutti_modelli_mean = tutti_modelli.mean(dim='time')
        tutti_modelli_std = tutti_modelli.std(dim='time')
        tutti_modelli_mean[f"{indicatore}_STD"] = tutti_modelli_std[indicatore]
        tutti_modelli_mean = tutti_modelli_mean.assign_coords({'lon': datasets_with_time[0].lon, 'lat': datasets_with_time[0].lat})

        output_file = f'{self.outputPath}/{self.modelData}_{indicatore.lower()}_{self.annoInizio}-{self.annoFine}_{self.annoInizioFuturo}-{self.annoFineFuturo}_{self.scenario_dir}.nc'
        tutti_modelli_mean.to_netcdf(output_file, mode='w', format="NETCDF4")

        return anomalia
    
    def su(self, thresh=25, freq='1YE', indicatore='SU'):
        self.hw(thresh=thresh, freq=freq, indicatore=indicatore, variabile='tasmax')

    def pet(self, indicatore='PET_TH'):
        scenario = self.scenario.replace(' ', '').replace('.', '')
        model_list = self.getNetcdfModelData(variabile='tas')
        anomalie = []
        
        for modello in model_list:
            regex = r"model([0-9]+)_rcp"
            matches = re.finditer(regex, modello, re.DOTALL)
            #match= enumerate(matches, start=1)
            modelloNumero = [match.group(1) for match in matches]
            modelloNumero = str(modelloNumero[0])
            print(f"Elaborazione modello: {modello}")
            print(modelloNumero)
            try:
                des_tas = self.getNetcdfOpenDataset(modello, variabile='tas')
                
                # print(f"Dimensioni iniziali des_tas: {des_tas.dims}")
                # print(f"Coordinate des_tas: {des_tas.coords}")
                
                # Salviamo le coordinate lon/lat originali
                original_lon = des_tas.lon
                original_lat = des_tas.lat
                
                if 'y' in des_tas.dims and 'x' in des_tas.dims:
                    try:
                        des_tas = des_tas.rename({'y': 'rlat', 'x': 'rlon'})
                    except:
                        print('no')
                pet_raw = xclim.indices.potential_evapotranspiration(
                    method='TW48', 
                    tas=des_tas
                )
                
                # print(f"Dimensioni pet_raw: {pet_raw.dims}")
                # print(f"Shape pet_raw: {pet_raw.shape}")
                
                # Riordiniamo le dimensioni per avere time come prima dimensione
                pet_raw = pet_raw.transpose('time', 'rlat', 'rlon')
                
                # Creazione dataset con dimensioni corrette e coordinate originali
                ds_pet = xr.Dataset({
                    indicatore: (['time', 'rlat', 'rlon'], pet_raw.values),
                    'lon': (['rlat', 'rlon'], original_lon.values),
                    'lat': (['rlat', 'rlon'], original_lat.values)
                },
                coords={
                    'time': pet_raw.time,
                    'rlat': pet_raw.rlat,
                    'rlon': pet_raw.rlon
                })
                
                # print(f"Dimensioni ds_pet: {ds_pet.dims}")
                # print(f"Shape ds_pet[{indicatore}]: {ds_pet[indicatore].shape}")
            
                file_indicatore_completo = f'{self.path_indicatori_cache}/{self.modelData}_{indicatore.lower()}_MODEL{modelloNumero}_{self.scenario_dir}.nc'
                ds_pet.to_netcdf(file_indicatore_completo, mode='w', format='NETCDF4')
                
                # Calcolo variazione
                variazione = self.crea_anomalia_percentuale(ds_pet)
                # Non serve più assegnare le coordinate qui dato che sono già nel dataset
                anomalie.append(variazione)
                
            except Exception as e:
                print(f"Errore nell'elaborazione del modello {modello}")
                print(f"Errore dettagliato: {str(e)}")
                print("Stack trace completo:")
                import traceback
                print(traceback.format_exc())
                continue
        
        if anomalie:
            anomalie_concat = xr.concat(anomalie, dim='model')
            tutti_modelli_mean = anomalie_concat.mean(dim='model')
            tutti_modelli_std = anomalie_concat.std(dim='model')
            tutti_modelli_mean[f"{indicatore}_STD"] = tutti_modelli_std[indicatore]
            tutti_modelli_mean = tutti_modelli_mean.assign_coords({'lon': anomalie[0].lon, 'lat': anomalie[0].lat})

            output_file = f'{self.outputPath}/{self.modelData}_{indicatore.lower()}_{self.annoInizio}-{self.annoFine}_{self.annoInizioFuturo}-{self.annoFineFuturo}_{scenario}.nc'
            tutti_modelli_mean.to_netcdf(output_file, mode='w', format='NETCDF4')
            return tutti_modelli_mean
        else:
            raise ValueError("Nessun modello elaborato con successo")
        
    
    def prcptot(self, indicatore='PRCPTOT'):
        model_list = self.getNetcdfModelData(variabile='TOT_PREC')
        anomalie = []
        lista_completa = []
        for modello in model_list:
            print(modello)
            miomodello = self._prcptot(freq='YS', indicatore='PRCPTOT', model=modello)
            lista_completa.append(miomodello)
            
        # aprire tutti i dataset 
        datasets = [xr.open_dataset(file) for file in lista_completa]
        datasets_with_time = [ds.expand_dims(time=[i]) for i, ds in enumerate(datasets)]
        tutti_modelli = xr.concat(datasets_with_time, dim='time')
        tutti_modelli = tutti_modelli.assign_coords({'lon': datasets_with_time[0].lon, 'lat': datasets_with_time[0].lat})
        # fare la media e std
        tutti_modelli_mean = tutti_modelli.mean(dim='time')
        tutti_modelli_std = tutti_modelli.std(dim='time')
        tutti_modelli_mean[f"{indicatore}_STD"] = tutti_modelli_std[indicatore]
        output_file = f'{self.outputPath}/{self.modelData}_prcptot_{self.annoInizio}-{self.annoFine}_{self.annoInizioFuturo}-{self.annoFineFuturo}_{self.scenario_dir}.nc'
        tutti_modelli_mean.to_netcdf(output_file, mode='w', format="NETCDF4")
        #return model_list

    def _prcptot(self, freq='YS', indicatore='PRCPTOT', model=None):
        
        # cambia il nome con passato e futuro
        file_indicatore_completo = f'{self.path_indicatori_cache}/{indicatore.lower()}_1980-2070_{self.scenario_dir}.nc'
        numero_modello = '00'
        modelloNumero = 0
        if model!=None:
            regex = r"model([0-9]+)_rcp"
            matches = re.finditer(regex, model, re.DOTALL)
            #match= enumerate(matches, start=1)
            modelloNumero = [match.group(1) for match in matches]
            modelloNumero = str(modelloNumero[0])
            print(modelloNumero)
            #for matchNum, match in enumerate(matches, start=1):
            #print (match.group(1))
                
            precip_data = self.getNetcdfOpenDataset(model, variabile='pr')
            try:
                
                precip_data = precip_data.rename({'y': 'rlat', 'x': 'rlon'})
            except:
                print('no')
            
            file_indicatore_completo = f'{self.path_indicatori_cache}/{self.modelData}_{indicatore.lower()}_MODEL{modelloNumero}_{self.scenario_dir}.nc'
        else:
            precip_data = self.getNetcdfModelData(variabile='TOT_PREC')
        #print(precip_data)
        # setto a 0 il rumore del modello
        print(f'Controllo i valori min/max prima di eliminare il rumore del modello')
        print(f'PRIMA Min:{precip_data.values.min()} Max:{precip_data.values.max()}')
        precip_data = xr.where(precip_data < 1, 0, precip_data)
        print(f'DOPO  Min:{precip_data.values.min()} Max:{precip_data.values.max()}')
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

        anomalia = self.crea_anomalia_percentuale(prcptot_dataset)
        anomalia.to_netcdf(file_indicatore_completo, mode='w', format="NETCDF4")
        #print(anomalia)
        return file_indicatore_completo
        # print(cdd_dataset)
        
        
        # output_file = f'{self.outputPath}/{self.modelData}_{indicatore.lower()}_{self.annoInizio}-{self.annoFine}_{self.annoInizioFuturo}-{self.annoFineFuturo}_{self.scenario_dir}.nc'
        # anomalia.to_netcdf(output_file, mode='w', format="NETCDF4")
        # return anomalia

    def su95p(self, thresh=29.2, freq='1YE', indicatore='SU95P'):
        self.hw(thresh=thresh, freq=freq, indicatore=indicatore)

    def tn5prctile(self, indicatore='TN5prctile', freq='YS'):
        """
        TN5prctile - 5° percentile della temperatura minima (°C)

        5° percentile della temperatura minima giornaliera.
        """
        self.txprctile(indicatore=indicatore, percentile=5, freq=freq, variabile='tasmin')
        
    def tn10prctile(self, indicatore='TN10prctile', freq='YS'):
        """
        TN5prctile - 10° percentile della temperatura minima (°C)

        10° percentile della temperatura minima giornaliera.
        """
        self.txprctile(indicatore=indicatore, percentile=10, freq=freq, variabile='tasmin')
        
    def tg95prctile(self, indicatore='TG95prctile', freq='YS'):
        """
        TG95prctile - 95° percentile della temperatura media (°C)

        95° percentile della temperatura massima giornaliera.
        """
        self.txprctile(indicatore=indicatore, percentile=95, freq=freq, variabile='tas')

    def tg99prctile(self, indicatore='TG99prctile', freq='YS'):
        """
        TG99prctile - 99° percentile della temperatura media (°C)

        99° percentile della temperatura massima giornaliera.
        """
        self.txprctile(indicatore=indicatore, percentile=99, freq=freq, variabile='tas')

    def txprctile(self, indicatore='TXprctile', percentile=99, freq='YS', variabile='scrivi la variabile'):
        """
        TX99prctile - 99° percentile della temperatura massima (°C)

        99° percentile della temperatura massima giornaliera.
        """

        model_list = self.getNetcdfModelData(variabile)
        lista_completa = []
        #self.tprctile(indicatore=indicatore, variabile='tasmax', percentile=99, freq=freq)
        #model_list = model_list[1:2]
        #print(model_list)
        for modello in model_list:
            print(f"Processando modello: {modello}")
            try:
                # Carica i dati e rinomina le coordinate
                data = self.getNetcdfOpenDataset(modello, variabile=variabile)
                try:
                    data = data.rename({'y': 'rlat', 'x': 'rlon'})
                except:
                    print('no')
                if variabile=='tas' or variabile=='tasmax' or variabile=='tasmin':
                    # trasformiamo da gradi kelvin in gradi celsius
                    miomodello = data - 273.15

                if variabile=='pr':
                    # elimino il rumore del modello
                    miomodello = xr.where(data < 1, 0, data)

                # periodi
                passato = miomodello.sel(time=slice(str(self.annoInizio) + '-01-01', str(self.annoFine) + '-12-31'))
                futuro = miomodello.sel(time=slice(str(self.annoInizioFuturo) + '-01-01', str(self.annoFineFuturo) + '-12-31'))
                
                # calcolo il percentile
                print('Calcolo il percentile per il modello')
                passato = self.calcolaPercentile(passato, percentile)
                futuro = self.calcolaPercentile(futuro, percentile)
                
                # calcolo anomalia
                anomalia = futuro - passato
                
                lista_completa.append(anomalia)
            except Exception as e:
                print(f"Errore nel processare il modello {modello}: {e}")
                continue

        # Carica tutti i dataset processati
        #datasets = [xr.open_dataset(file) for file in lista_completa]
        
        # Aggiungi dimensione temporale per l'ensemble
        datasets_with_time = [ds.expand_dims(time=[i]) for i, ds in enumerate(lista_completa)]
        
        #Calcola la media dell'ensemble
        tutti_modelli = xr.concat(datasets_with_time, dim='time')
        tutti_modelli_std = tutti_modelli.std(dim='time')
        tutti_modelli = tutti_modelli.mean(dim='time')
        
        # tutti_modelli[f"{indicatore}_STD"] = tutti_modelli_std
        #tutti_modelli.assign(variable_name=(['STD'],tutti_modelli_std))
        # Aggiunta delle coordinate geografiche
        tutti_modelli = tutti_modelli.assign_coords({
            'lon': lista_completa[0].lon, 
            'lat': lista_completa[0].lat
        })
        tutti_modelli = tutti_modelli.rename(indicatore.upper())
        ensamble=tutti_modelli.to_dataset()
        #print(tutti_modelli)
        ensamble[f"{indicatore.upper()}_STD"] = tutti_modelli_std
        #print(tutti_modelli)
        #print(tutti_modelli)
        # rinomino la variabile con il nome indicatore
        
        #print(ensamble)
        # ensamble = tutti_modelli.rename({'tasmin': indicatore})
        # ensamble = tutti_modelli.rename({'tas': indicatore})
        # ensamble = tutti_modelli.rename({'tasmax': indicatore})
        print("Completo")
         # Salva il risultato finale
        output_file = f'{self.outputPath}/{self.modelData}_{indicatore.lower()}_{self.annoInizio}-{self.annoFine}_{self.annoInizioFuturo}-{self.annoFineFuturo}_{self.scenario_dir}.nc'
        ensamble.to_netcdf(output_file, mode='w', format="NETCDF4")
        return ensamble
    
    def tx90prctile(self, indicatore='TX90prctile', freq='YS', variabile='tasmax'):
        """
        TX90prctile - 90° percentile della temperatura massima (°C)

        90° percentile della temperatura massima giornaliera.
        """
        self.txprctile(indicatore=indicatore, percentile=90, freq=freq, variabile=variabile)

    def tx95prctile(self, indicatore='TX95prctile', freq='YS', variabile='tasmax'):
        """
        TX95prctile - 95° percentile della temperatura massima (°C)

        95° percentile della temperatura massima giornaliera.
        """
        self.txprctile(indicatore=indicatore, percentile=95, freq=freq, variabile=variabile)

    def tx99prctile(self, indicatore='TX99prctile', freq='YS', variabile='tasmax'):
        """
        TX99prctile - 99° percentile della temperatura massima (°C)

        99° percentile della temperatura massima giornaliera.
        """
        self.txprctile(indicatore=indicatore, percentile=99, freq=freq, variabile=variabile)

    def pr95prctile(self, indicatore='PR95prctile', freq='YS'):
        """
        PR95prctile - 95° percentile della precipitazione (mm)

        95° percentile della precipitazione giornaliera.
        """
        self.txprctile(indicatore=indicatore, percentile=95, freq=freq, variabile='pr')

    def pr99prctile(self, indicatore='PR99prctile', freq='YS'):
        """
        PR95prctile - 99° percentile della precipitazione (mm)

        99° percentile della precipitazione giornaliera.
        """
        self.txprctile(indicatore=indicatore, percentile=99, freq=freq, variabile='pr')

    
    def _r10(self, freq='1Y', thresh=10, indicatore='R10', modello=None):
        pr = self.getNetcdfOpenDataset(modello, 'pr')
        try:
            pr = pr.rename({'y': 'rlat', 'x': 'rlon'})
        except:
            print('no')
        #elimino il rumore
        miomodello = xr.where(pr < 1, 0, pr)
        # assegno l'unità di misura
        miomodello = miomodello.assign_attrs(units='mm/day')

        indicatore_tot = xr.where(miomodello > thresh, 1, 0)

        indicatore_tot = indicatore_tot.resample(time=freq).sum(dim='time')

        dataset = xr.Dataset({
            indicatore: (['time', 'rlat', 'rlon'], indicatore_tot.values),
            'time': indicatore_tot.time,  # Usa l'intero oggetto time, non solo l'anno
            'rlat': indicatore_tot.rlat,
            'rlon': indicatore_tot.rlon
        })

        # aggiungo le coordinate
        dataset = dataset.assign_coords({'lon': indicatore_tot.lon, 'lat': indicatore_tot.lat})

        
        return dataset

    def r10(self, freq='1Y', thresh=10, indicatore='R10'):
        model_list = self.getNetcdfModelData('pr')
        
        lista_completa = []

        for modello in model_list:
            print(f"Processando modello: {modello}")
            # calcolo l'indicatore per il modello
            dataset = self._r10(freq=freq, thresh=thresh, indicatore=indicatore, modello=modello)
            
            # calcolo l'anomalia
            anomalia = self.crea_anomalia_abs(dataset)

            lista_completa.append(anomalia)

        tutti_modelli = xr.concat(lista_completa, dim='time')
        tutti_modelli_std = tutti_modelli.std(dim='time')
        tutti_modelli = tutti_modelli.mean(dim='time')
        
        # Aggiunta delle coordinate geografiche
        tutti_modelli = tutti_modelli.assign_coords({
            'lon': lista_completa[0].lon, 
            'lat': lista_completa[0].lat
        })

        #tutti_modelli = tutti_modelli.rename(indicatore.upper())

        ensamble=tutti_modelli
        ensamble[f"{indicatore.upper()}_STD"] = tutti_modelli_std[indicatore]

        
        # Salva il risultato finale
        output_file = f'{self.outputPath}/{self.modelData}_{indicatore.lower()}_{self.annoInizio}-{self.annoFine}_{self.annoInizioFuturo}-{self.annoFineFuturo}_{self.scenario_dir}.nc'
        print("Completo")
        print(output_file)
        ensamble.to_netcdf(output_file, mode='w', format="NETCDF4")
        return ensamble

    def r20(self):
        self.r10(freq='1Y', thresh=20, indicatore='R20')

    def r50(self):
        self.r10(freq='1Y', thresh=50, indicatore='R50')
        
    def rxday(self, indicatore='RXDAY', thresh=1, freq='YS'):
        model_list = self.getNetcdfModelData('pr')
        lista_completa = []
        #self.tprctile(indicatore=indicatore, variabile='tasmax', percentile=99, freq=freq)
        #model_list = model_list[1:2]
        #print(model_list)
        for modello in model_list:
            print(f"Processando modello: {modello}")
            try:
                # Carica i dati e rinomina le coordinate
                data = self.getNetcdfOpenDataset(modello, variabile='pr')
                try:
                    data = data.rename({'y': 'rlat', 'x': 'rlon'})
                except:
                    print('no')
                miomodello = xr.where(data < 1, 0, data)
                miomodello = miomodello.assign_attrs(units='mm/day')
                indicatore_tot = xclim.indices.max_n_day_precipitation_amount(miomodello, window=thresh, freq=freq)

                dataset = xr.Dataset({
                    indicatore: (['time', 'rlat', 'rlon'], indicatore_tot.values),
                    'time': indicatore_tot.time,  # Usa l'intero oggetto time, non solo l'anno
                    'rlat': indicatore_tot.rlat,
                    'rlon': indicatore_tot.rlon
                })

                # aggiungo le coordinate
                dataset = dataset.assign_coords({'lon': indicatore_tot.lon, 'lat': indicatore_tot.lat})

                # periodi
                passato = dataset.sel(time=slice(str(self.annoInizio) + '-01-01', str(self.annoFine) + '-12-31')).mean(dim='time')
                futuro = dataset.sel(time=slice(str(self.annoInizioFuturo) + '-01-01', str(self.annoFineFuturo) + '-12-31')).mean(dim='time')
                
                # calcolo anomalia
                anomalia = ((futuro - passato)/passato)*100
                
                lista_completa.append(anomalia)
            
            except Exception as e:
                print(f"Errore nel processare il modello rxday: {modello}: {e}")
                continue

        # Aggiungi dimensione temporale per l'ensemble
        #datasets_with_time = [ds.expand_dims(time=[i]) for i, ds in enumerate(lista_completa)]
        
        # Calcola la media dell'ensemble
        tutti_modelli = xr.concat(lista_completa, dim='time')
        tutti_modelli_std = tutti_modelli.std(dim='time')
        tutti_modelli = tutti_modelli.mean(dim='time')
        
        # Aggiunta delle coordinate geografiche
        tutti_modelli = tutti_modelli.assign_coords({
            'lon': lista_completa[0].lon, 
            'lat': lista_completa[0].lat
        })
        
        # rinomino la variabile con il nome indicatore
        
        #ensamble = tutti_modelli.rename(indicatore.upper())
        ensamble=tutti_modelli
        ensamble[f"{indicatore.upper()}_STD"] = tutti_modelli_std[indicatore]
        
         # Salva il risultato finale
        output_file = f'{self.outputPath}/{self.modelData}_{indicatore.lower()}_{self.annoInizio}-{self.annoFine}_{self.annoInizioFuturo}-{self.annoFineFuturo}_{self.scenario_dir}.nc'
        print(f"Completo {indicatore}")
        print(output_file)
        ensamble.to_netcdf(output_file, mode='w', format="NETCDF4")
    
    def rx1day(self, indicatore='RX1DAY'):
        self.rxday(indicatore='RX1DAY', thresh=1, freq='YS')

    def rx3day(self, indicatore='RX3DAY'):
        self.rxday(indicatore=indicatore, thresh=3, freq='YS')

    def rx5day(self, indicatore='RX5DAY'):
        self.rxday(indicatore=indicatore, thresh=5, freq='YS')

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
    
    def spei3(self, indicatore = 'SPEI3', scale=3):
        file_indicatore_completo = f'{self.outputPath}/{self.modelData}_{indicatore.lower()}_{self.annoInizio}-{self.annoFine}_{self.annoInizioFuturo}-{self.annoFineFuturo}_{self.scenario_dir}.nc'
        
        model_list = self.getNetcdfModelData('pr')
        # model_list = model_list[0:2]
        lista_completa = []
        
        # annoInizio = 1981
        # annoInizioFuturo = 2036
        # annoFine = annoInizio + 30 - 1
        # annoFineFuturo = annoFine + 30 - 1

        cal_start = '1981-01-01'
        cal_end='2010-12-01'

        # outputPath = '/work/cmcc/ab03024/indicatori/BOSNIA_CORDEX'

        spi_list = []
        for modello in model_list:
            print(f"Processando modello: {modello}")
            regex = r"model([0-9]+)_rcp"
            matches = re.finditer(regex, modello, re.DOTALL)
            modelloNumero = [match.group(1) for match in matches][0]
            
            tp_daily = xr.open_dataset(modello)
            #tp_daily=tp_daily['pr']
            daily_water_budget = f'{self.path_indicatori_cache}/EU-CORDEX-11_DWB_MODEL{modelloNumero}_1981-2100_{self.scenario_dir}.nc'
            daily_water_budget = xr.open_dataset(daily_water_budget)
            spei = self.drought_spei(daily_water_budget, scale=scale)
            print(spei.isel(time=5).values)
            # spei = spei.rename({'y': 'rlat', 'x': 'rlon'})
            
            classes = [(-2, float('-inf'), 'Extremely_dry'),
                        (-1.5, -2, 'Severely_dry'),
                        (-1, -1.5, 'Moderately_dry'), 
                        (1, -1, 'Near_normal'), 
                        (1.5, 1, 'Moderately_wet'), 
                        (2, 1.5, 'Very_wet'),
                        (float('inf'), 2, 'Extremely_wet')]

            #numero di tempi = 12 mesi *30 anni=360 mesi
            numero_di_tempi = 12*30
            ds = xr.Dataset()
            for thresholds in classes:
                spei_one_class = spei['drought'].where((spei['drought'] < thresholds[0]) & (spei['drought'] > thresholds[1]))
                
                spei_benchmark = spei_one_class.sel(time = slice(cal_start, cal_end))
                spei_scenario = spei_one_class.sel(time = slice(str(self.annoInizioFuturo) + '-01-01', str(self.annoFineFuturo) + '-12-01'))
                
                percentage_future = spei_scenario.count(dim='time') / len(spei_scenario.time) * 100
                percentage_past = spei_benchmark.count(dim='time') / len(spei_benchmark.time) * 100

                spei_anomaly = percentage_future - percentage_past
                
                ds[thresholds[2]] = spei_anomaly
            lista_completa.append(ds)
        # concateno
        spei = xr.concat(lista_completa, dim='modello')
        
        print('ensamble')
        spei_mean = spei.mean(dim='modello')
        # Calcolo di altre misure statistiche se necessarie
        spei_std = spei.std(dim='modello')
        #indiVar = f"{indicatore}_STD"
        #spei_mean = spei_mean.assign(indiVar = spei_std)
        #spei_mean[f"{indicatore}_STD"] = spei_std[indicatore]
        
        for thresholds in classes:
            spei_mean[f"{thresholds[2]}_STD"] = spei_std[thresholds[2]]
        # Aggiunta delle coordinate geografiche
        pippo = spei_mean.assign_coords({
            'lon': lista_completa[0].lon, 
            'lat': lista_completa[0].lat
        })
        print(file_indicatore_completo)
        pippo.to_netcdf(file_indicatore_completo, mode='w', format="NETCDF4")
        
        return spei
    def spei6(self):
        return self.spi3(indicatore = 'SPEI6', scale=6)
    
    def spei12(self):
        return self.spi3(indicatore = 'SPEI12', scale=12)
    
    def spei24(self):
        return self.spi3(indicatore = 'SPEI24', scale=24)
        
    def drought_spi(self, tp_daily, baseline = (1981, 2011), scale=3):
        from climate_indices import indices as cindx
        """ Using Standard Precipitation index """
        
        ## total monthly precipitation
        #tp_monthly = tp_daily.resample(time='MS').sum(dim='time')
        #tp_monthly = tp_daily.resample(time='MS').sum()  # Riga corretta
        tp_monthly = self.resampleMonth(tp_daily)
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
    
    def spi3(self, indicatore = 'SPI3', scale=3):
        file_indicatore_completo = f'{self.outputPath}/{self.modelData}_{indicatore.lower()}_{self.scenario_dir}.nc'
        
        model_list = self.getNetcdfModelData('pr')
        # model_list = model_list[0:2]
        lista_completa = []
        
        # annoInizio = 1981
        # annoInizioFuturo = 2036
        # annoFine = annoInizio + 30 - 1
        # annoFineFuturo = annoFine + 30 - 1

        cal_start = '1981-01-01'
        cal_end='2010-12-01'

        # outputPath = '/work/cmcc/ab03024/indicatori/BOSNIA_CORDEX'

        spi_list = []
        for modello in model_list:
            print(f"Processando modello: {modello}")
            regex = r"model([0-9]+)_rcp"
            matches = re.finditer(regex, modello, re.DOTALL)
            modelloNumero = [match.group(1) for match in matches][0]
            
            tp_daily = xr.open_dataset(modello)
            #tp_daily=tp_daily['pr']
            
            spi = self.drought_spi(tp_daily, scale=scale)
            try:
                
                spi = spi.rename({'y': 'rlat', 'x': 'rlon'})
            except:
                print('no')
            
            classes = [(-2, float('-inf'), 'Extremely_dry'),
                        (-1.5, -2, 'Severely_dry'),
                        (-1, -1.5, 'Moderately_dry'), 
                        (1, -1, 'Near_normal'), 
                        (1.5, 1, 'Moderately_wet'), 
                        (2, 1.5, 'Very_wet'),
                        (float('inf'), 2, 'Extremely_wet')]

            #numero di tempi = 12 mesi *30 anni=360 mesi
            numero_di_tempi = 12*30
            ds = xr.Dataset()
            for thresholds in classes:
                spi_one_class = spi['drought'].where((spi['drought'] < thresholds[0]) & (spi['drought'] > thresholds[1]))
                
                spi_benchmark = spi_one_class.sel(time = slice(cal_start, cal_end))
                spi_scenario = spi_one_class.sel(time = slice(str(self.annoInizioFuturo) + '-01-01', str(self.annoFineFuturo) + '-12-01'))
                
                percentage_future = spi_scenario.count(dim='time') / len(spi_scenario.time) * 100
                percentage_past = spi_benchmark.count(dim='time') / len(spi_benchmark.time) * 100

                spi_anomaly = percentage_future - percentage_past
                
                ds[thresholds[2]] = spi_anomaly
            lista_completa.append(ds)
        # concateno
        spi = xr.concat(lista_completa, dim='modello')
        # Prima calcolo la deviazione standard tra i modelli
        tutti_modelli_std = spi.std(dim='modello')
        # Poi calcolo la media
        spi_mean = spi.mean(dim='modello')
        
        # Aggiungo sia la media che la deviazione standard per ogni classe
        for classe in tutti_modelli_std.data_vars:
            # Aggiungo la std con il suffisso _DEV
            spi_mean[f"{classe}_STD"] = tutti_modelli_std[classe]
        
        # Aggiunta delle coordinate geografiche
        spi_mean = spi_mean.assign_coords({
            'lon': lista_completa[0].lon, 
            'lat': lista_completa[0].lat
        })
        
        spi_mean.to_netcdf(file_indicatore_completo, mode='w', format="NETCDF4")
        
    def spi6(self):
        return self.spi3(indicatore = 'SPI6', scale=6)
    
    def spi12(self):
        return self.spi3(indicatore = 'SPI12', scale=12)
    
    def spi24(self):
        return self.spi3(indicatore = 'SPI24', scale=24)
    
    def tr(self, thresh=20, freq='1YE', indicatore='TR'):
        '''
        Numero di giorni con temperatura minima giornaliera maggiore di 20°
        '''
        self.hw(thresh=thresh, freq=freq, indicatore=indicatore, variabile='tasmin')

    def tr_pr(self, indicatore='TRPR', freq='YS', t=0, mode='max'):
        
        model_list = self.getNetcdfModelData('pr')
        
        lista_completa = []
        #self.tprctile(indicatore=indicatore, variabile='tasmax', percentile=99, freq=freq)
        model_list = model_list[1:2]
        #print(model_list)
        for modello in model_list:
            print(f"Processando modello: {modello}")
            try:
                # Carica i dati e rinomina le coordinate
                pr = self.getNetcdfOpenDataset(modello, variabile='pr')
                try:
                    pr = pr.rename({'y': 'rlat', 'x': 'rlon'})
                except: 
                    print('no')
                
                # elimino il rumode del modello
                pr = xr.where(pr < 1, 0, pr)
                pr = pr.assign_attrs(units='mm/day')
                pr_past = pr.sel(time = slice(str(self.annoInizio) + '-01-01', str(self.annoFine) + '-12-01'))
                tr_past_raw = frequency_analysis(pr_past, t=t, dist='genextreme', mode=mode, freq=freq)
                tr_past = xr.Dataset({
                    indicatore: (['rlat', 'rlon'], tr_past_raw.values),
                    'lat': (['rlat', 'rlon'], pr.lat.values),
                    'lon': (['rlat', 'rlon'], pr.lon.values),
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
    
                #calcolo anomalia in percentuale
                tr_anomaly_raw = (tr_future[indicatore] - tr_past[indicatore]) / tr_past[indicatore] * 100
                
                del tr_past
                tr_anomaly = xr.Dataset({
                    indicatore: (['rlat', 'rlon'], tr_anomaly_raw.values),
                    'lat': (['rlat', 'rlon'], tr_future.lat.values),
                    'lon': (['rlat', 'rlon'], tr_future.lon.values),
                    #'return_period': tr_anomaly_raw.return_period,
                    'rlat': tr_anomaly_raw.rlat,
                    'rlon': tr_anomaly_raw.rlon})
                
                lista_completa.append(tr_anomaly)
            
            except Exception as e:
                print(f"Errore nel processare il modello rxday: {modello}: {e}")
                
        tutti_modelli = xr.concat(lista_completa, dim='time')
        tutti_modelli_mean = tutti_modelli.mean(dim='time')
        tutti_modelli_std = tutti_modelli.std(dim='time')
        tutti_modelli_mean[f"{indicatore}_STD"] = tutti_modelli_std[indicatore]
        
        # Aggiunta delle coordinate geografiche
        tutti_modelli_mean = tutti_modelli_mean.assign_coords({
            'lon': lista_completa[0].lon, 
            'lat': lista_completa[0].lat
        })
        print(tutti_modelli_mean)
        # rinomino la variabile con il nome indicatore
        #ensamble = tutti_modelli.rename(indicatore.upper())
        # print(ensamble)
        # ensamble = tutti_modelli.rename({'tasmin': indicatore})
        # ensamble = tutti_modelli.rename({'tas': indicatore})
        # ensamble = tutti_modelli.rename({'tasmax': indicatore})
        
         # Salva il risultato finale
        output_file = f'{self.outputPath}/{self.modelData}_{indicatore.lower()}_{self.annoInizio}-{self.annoFine}_{self.annoInizioFuturo}-{self.annoFineFuturo}_{self.scenario_dir}.nc'
        print(f"Completo {indicatore}")
        print(output_file)
        tutti_modelli_mean.to_netcdf(output_file, mode='w', format="NETCDF4")
        return tutti_modelli_mean
    
    def tr20pr(self, indicatore='TR20PR', freq='YS', t=20, mode='max'):
        return self.tr_pr(indicatore=indicatore, freq=freq, t=t, mode=mode)
    
    def tr50pr(self, indicatore='TR50PR', freq='YS', t=50, mode='max'):
        return self.tr_pr(indicatore=indicatore, freq=freq, t=t, mode=mode)

    def tr100pr(self, indicatore='TR100PR', freq='YS', t=100, mode='max'):
        return self.tr_pr(indicatore=indicatore, freq=freq, t=t, mode=mode)

