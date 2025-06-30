! (C) Copyright 1996- ECMWF.
!
! This software is licensed under the terms of the Apache Licence Version 2.0
! which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
! In applying this licence, ECMWF does not waive the privileges and immunities
! granted to it by virtue of its status as an intergovernmental organisation nor
! does it submit to any jurisdiction.

!> @brief GEFF runtime. Code structure:
!>
!> A. set-up conditions
!>   1. weather type
!>   2. fuel model based on the JRC climatological maps
!>   3. climate class
!>   4. vegetation stage
!>   5. mean slope
!>
!> B. NFDRS
!>   1. calculate moisture/loading content for the various fuel models (prognostic part)
!>     1. 1hr fuel
!>     2. 10hr fuel
!>     3. 100hr fuel
!>     4. 1000hr fuel
!>     5. herbaceous and wood
!>   2. Fire Model indices (diagnostic part)
!>     1. Fire characteristics /properties (fire_prop)
!>       1. Spread component
!>       2. Energy Release
!>       3. Burning Index
!>     2. Fire occurrence probability (fire_prob)
!>       1. ignition probability
!>       2. Human caused fire occurrence index
!>       3. Lightning-caused fire occurrence index
!>       4. Fire load index
!>     3. Mask outputs for wet/snow/dew  conditions
!>
!> C. MARK-5
!>
!> D. FWI
!>   1. The fine fuel moisture code
!>   2. The duff moisture /content code
!>   3. The Drought Code
!>   4. Initial Spread Index
!>   5. Build-up Index
!>
!> E. output
!
!> @author Di Giuseppe, F., ECMWF
!> @author Maciel, P., ECMWF
PROGRAM geff

  USE mo_fire
  USE mo_fwi
  USE mo_io_eccodes
  USE mo_utilities
  USE mo_version
  USE mo_control
  USE, intrinsic :: ieee_arithmetic
  IMPLICIT NONE

  ! local variables
  INTEGER :: istep,icheck,idate=0,i
  INTEGER(4) :: actualdate
  INTEGER(4) :: restartdate

  ! variables for prognostic calculations
  REAL :: zrain, ztemp &
&           ,zwspeed, zdp&
&           ,zlat
  DOUBLE PRECISION :: zrh

  ! variables for diagnostic calculations
  REAL :: mo, kd, kw, &
       & mr, mr0, mr1,&
       & ed, ed0, ed1, ed2,&
       & ko, ko0, ko1, ko2,&
       & ew, ew0, ew1, ew2,&
       & kl, kl0, kl1, kl2,&
       & vv,rd, qo, qr,dr,&
       & re, moo, mrr, bb, pr, k,&
       & m, fd, fwiB,ff, &
       & dl,lf,fwind,uu,mm,t
  DOUBLE PRECISION :: rf

  CHARACTER (len=8) :: str_date
  CHARACTER (len=4) :: str_time

  ! local integer scalars
  INTEGER :: jyear,jmonth,jday,jhh
  INTEGER :: ndaydiag = 1          ! diagnostics every n days

  ! fortran timer functions
  REAL :: time1=0.0,time2=0.0
  LOGICAL :: ltimer=.false.      ! turn the cpu timer on for the first timestep

NAMELIST /control/ output_file, output_restart, output_constant, inidate, initime, dt, restart_file, restart_day, now
NAMELIST /climate/ tempfile,maxtempfile,mintempfile,rhfile,maxrhfile,minrhfile,rainfile,ccfile,wspeedfile,snowfile,dpfile,vsfile
NAMELIST /constdata/ rainclimfile, lsmfile, crfile, fmfile, cvfile, slopefile


  ! -----
  ! SETUP
  ! -----

  !--------------------------
  ! get basic run info

  PRINT *, '-------------- GEFF ---------------'
  PRINT *, 'version: ', version
  PRINT *, 'file: ', "'"//namelst//"'"

  OPEN (8, file=namelst, status='OLD')
  READ (8, nml=control)
  READ (8, nml=climate)
  READ (8, nml=constdata)
  CLOSE (8)

  WRITE (str_date, '(I8)') inidate
  WRITE (str_time, '(I4)') initime

  PRINT *, 'now (reference date): ', now
  PRINT *, 'dt (hours): ', dt
  PRINT *, "time units: ", "hours since "//str_date(1:4)//"-"//str_date(5:6)//"-"//str_date(7:8)//" &
  & "//str_time(1:2)//":"//str_time(3:4)//" UTC"

  PRINT *, "Initialize..."
  CALL io_initialize
  PRINT *, "Initialize... DONE"


  ! -----
  ! START
  ! -----

  ! define the date at which to dump the restart
  restartdate = add_day(inidate, initime, (restart_day-1)*24)
  PRINT *, "Restart date:", restartdate, "Restart time:", initime

  DO istep=1,ntimestep

     IF (ltimer) CALL timer('go  ',icheck,time1,time2) !set up timer

     actualdate = add_day(inidate, initime, nhours(istep))

     jyear=INT(actualdate/10000.)
     jmonth=INT((actualdate-jyear*10000.)/100.)
     jday=INT((actualdate-jyear*10000.)-jmonth*100.)
     jhh=initime+MOD((nhours(istep)),24)

     PRINT*,'step ',istep, "actualdate", actualdate, "time (hours)",jhh

     IF (MOD(istep,ndaydiag)==0) PRINT *, 'date ',idate,"actualdate", actualdate

     ! read in met data timeslice
     !---------------------------
     CALL io_getdata(istep)

     !-------------
     ! GRIDDED LOOP
     !-------------

     DO i = 1, npoints

           zrain=MAX(rrain(i),0.0) ! 1 day rainfall
           ztemp=MAX(rtemp(i),0.0) ! temperature on a daily timestep
           zrh=MIN(MAX(rrh(i),0.0), 100.0) ! relative humidity
           zwspeed=MAX(rwspeed(i),0.0) ! wind speed
           zlat=lats(i)                                                ! index the ammount of annual


           !---------------------------------------
           ! ONLY  MODEL POINT IF
           ! NOT A 100%  LAKE/SEA POINT
           ! YES ALSO  ARCTIC CLIMATE .AND. icr(i) .gt.  0
           ! fuel model IS DEFINED (i.e. one of the  20 valid fuel models)
           ! vegetation stage is defined  (this has been removed )
           !---------------------------------------

           ! calculations is performed only on land points for all the indices
           !
      IF (rlsm(i) .gt. 0.0001 .and. zlat .gt. -60.0 )  THEN
            ! 0- set-up conditions
   !---------------------------------------------------------------------------

!C)  FWI
!=======================================================================================================

!1 The fine fuel moisture code
!==========================================================================

!Parameters for the calculation:
!    TEMP is the 12:00 LST temperature in degrees celsius
!    RH is the 12:00 LST relative humidity in %
!    WIND is the 12:00 LST wind speed in kph
!    RAIN is the 24-hour accumulated rainfall in mm, calculated at 12:00 LST
!    FFMCPrev is the previous day's FFMC

!TEST ***************************
!test ffmc should be  87.692980092774448 if
!        ztemp=17.+r0CtoK
!        zrh=42.
!        zwspeed=6.944
!        zrain=0
!        fwi_risk(i)%ffmc=85
!TEST ***************************

        mo = 147.2 * (101.0 - fwi_risk(i)%ffmc)/(MAX((59.5 + fwi_risk(i)%ffmc),reps))
        IF ( zrain .GT. 0.5) THEN
           rf = zrain-0.5
           mr0 = EXP(-100.0 /(251.0 - mo))
           IF (rf < 0.01) THEN
            mr1 = 1
           ELSE
            mr1 = (1 - EXP(-6.93 / rf))
           ENDIF
           mr = mo + 42.5 * rf * mr0 * mr1

           IF (mo .GT. 150.0)  mr = mr+0.0015*(mo-150.0)**(2.0)*(rf)**(0.5)

           IF (mr .GT. 250.0)  mr = 250.0
           mo = mr
        ENDIF

        ed0 = zrh**( 0.679)
        ed1 = exp((zrh-100.0)/10.0)
        ed2 = 1.0-EXP(-0.115*zrh)
        ed = 0.942*ed0+11.0*ed1+0.18*(21.1-(ztemp-r0CtoK))*ed2

        IF (mo .GT. ed) THEN
           ko0 = 1-(zrh/100.0)**(1.7)
           ko1 = (zwspeed*tokmhr)**(0.5)
           ko2 = 1.0-(zrh/100.0)**(8.0)
           ko = 0.424*ko0 + 0.0694*ko1*ko2
           kd = ko*0.581*EXP(0.0365*(ztemp-r0CtoK))
           mm = ed+(mo-ed)*(10.0)**(-kd)
        ELSE
           ew0 = (zrh)**(0.753)
           ew1 = EXP((zrh-100.0)/10.0)
           ew2 = 1.0-EXP(-0.115*zrh)
           ew = 0.618*ew0 + 10*ew1 + 0.18*(21.1-(ztemp-r0CtoK))*ew2

           IF (mo .LT.  ew) THEN
              kl0 = 1-((100-zrh)/100)**(1.7)
              kl1 = (zwspeed*tokmhr)**(0.5)
              kl2 = 1-((100-zrh)/100)**(8)
              kl = 0.424 * kl0 + 0.0694 * kl1 * kl2
              kw = kl*0.581*EXP(0.0365*(ztemp-r0CtoK))
              mm = ew-(ew-mo)*(10.0)**(-kw)
           ELSE
              mm = mo
           ENDIF
        ENDIF
! rgw ffmc is scaled between 2% and 101%
        fwi_risk(i)%ffmc = MIN(MAX(59.5*(250.0-mm)/(147.2+mm),2.0),101.0)

!        WRITE (9,*) 'fwi_risk(i)%ffmc',fwi_risk(i)%ffmc,87.692980092774448
   !     m=MAX(m,0.0)

! 2  The Duff Moisture Code
!==========================================================================

!Parameters for the calculations and units:
!    TEMP is the 12:00 LST temperature in degrees celsius
!    RH is the 12:00 LST relative humidity in %
!    RAIN is the 24-hour accumulated rainfall in mm, calculated at 12:00 LST
!    DMC is the prevvious day's DMC
!    Lat is the latitude in decimal degrees of the location for which calculations are being made
!    Month is the month of Year (1..12) for the current day's calculations.'''
!
!TEST ***************************
!test dnc should be  8.5450511359999997  if
!        ztemp=17.+r0CtoK
!        zrh=42.
!        zrain=0
!        fwi_risk(i)%dmc= 6
!        zlat=45.98
!        jmonth=4
!TEST ***************************
   IF (zrain .GT. 1.5) THEN
      re = 0.92 * zrain - 1.27
      moo = 20.0 + EXP(5.6348 -fwi_risk(i)%dmc / 43.43)

      IF  (fwi_risk(i)%dmc .LE. 33.0) THEN
         bb = 100.0 / (0.5 + 0.3 * fwi_risk(i)%dmc)
       ELSE IF (fwi_risk(i)%dmc .LE. 65.0) THEN
         bb = 14.0 - 1.3 * LOG(fwi_risk(i)%dmc)
       ELSE
         bb = 6.2 * LOG(fwi_risk(i)%dmc) - 17.2
       ENDIF
       mrr = moo + 1000.0 * re / (48.77 + bb * re)

       pr = 244.72 - 43.43 * LOG(mrr - 20.0)

       IF ( pr .GT. 0.0) THEN
         fwi_risk(i)%dmc = pr
       ELSE
         fwi_risk(i)%dmc = 0.0
       ENDIF
     ENDIF

     IF ((ztemp-r0CtoK) .GT. -1.1) THEN

       CALL DayLength(zlat,jmonth,dl)

       k = 1.894 * ((ztemp-r0CtoK) + 1.1) * (100.0 - zrh) * dl * 0.000001
     ELSE
       k = 0.0
     ENDIF

  ! increments

     !DMC is bounded between 0 and 10,000 (DMC could get to infinity and it will get to very high values over desereted areas where no fires can happens as there is no fuel )
     fwi_risk(i)%dmc=MIN(MAX(fwi_risk(i)%dmc + 100.0 * k,0.0),10000.0)
!     WRITE (9,*) 'fwi_risk(i)%dmc',fwi_risk(i)%dmc,' 8.5450511359999997'
! 3  The Drought Code
!==========================================================================

!Parameters for the calculations and units:
!    TEMP is the 12:00 LST temperature in degrees celsius
!    RAIN is the 24-hour accumulated rainfall in mm, calculated at 12:00 LST
!    DMC is the prevvious day's DMC
!    Lat is the latitude in decimal degrees of the location for which calculations are being made
!    Month is the month of Year (1..12) for the current day's calculations.'''
!TEST ***************************
!test dc should be  19.013999999999999  if
!        ztemp=17.+r0CtoK
!        zrain=0
!        fwi_risk(i)%dc= 15
!        zlat=45.98
!        jmonth=4
!TEST ***************************

    IF (zrain .GT. 2.8) THEN
        rd = 0.83 * zrain - 1.27
      ! rd = 0.83 * zrain -  2.8
      Qo = MIN(800.0 * EXP(-fwi_risk(i)%dc / 400.0),800.0)
        Qr = Qo + 3.937 * rd
        Dr = 400.0 * LOG(800.0 / Qr)

        IF (Dr .GT. 0.0) THEN
            fwi_risk(i)%dc = Dr
         ELSE
            fwi_risk(i)%dc = 0.0
         ENDIF
       ENDIF

      CALL DryingFactor(zlat,jmonth,Lf)
      ! Evapotranspiration factors
      IF ((ztemp-r0CtoK) .GT. -2.8)THEN
        vv = (0.36 * ((ztemp-r0CtoK)+2.8) + Lf)/ 2.
      ELSE
        vv = Lf / 2.
      ENDIF
      IF (vv .LT. 0 ) THEN
        vv=0.0
      END IF
            !upper limit the DC as it can get to very large numbers that
      !are outside the limits of single precision
      !DC is bounded between 0 and 10,000

      fwi_risk(i)%dc=MIN(MAX(fwi_risk(i)%dc + vv,0.0),10000.0)

!       WRITE (9,*) 'fwi_risk(i)%dc',fwi_risk(i)%dc,'19.013999999999999 '
! 4   Initial Spread Index
!==========================================================================
!
!    '''Calculates today's Initial Spread Index
!Parameters:
!    WIND is the 12:00 LST wind speed in kph
!    FFMC is the current day's FFMC'''

!TEST ***************************
! test ISI should be  10.853661073655068   if
!
!        zwspeed=6.944
!        fwi_risk(i)%ffmc=87.692980092774448
!TEST ***************************

    fWIND = EXP(0.05039*zwspeed*tokmhr)

    m = 147.2 * (101.0 - fwi_risk(i)%ffmc) /(59.5 + fwi_risk(i)%ffmc)

    ff = 91.9 *EXP(-0.1386 * m) * (1.0 + m**(5.31) / 49300000.0)

    fwi_risk(i)%isi= 0.208 * fWIND * ff


!     WRITE (9,*) 'fwi_risk(i)%isi',fwi_risk(i)%isi,'10.85366107365 '

! 5    Buildup Index
!==========================================================================
!    '''Calculates today's Buidup Index
!Parameters:
!    DMC is the current day's Duff Moisture Code
!    DC is the current day's Drought Code'''
!TEST ***************************
! test BUI should be  8.4904265358371838   if
!      fwi_risk(i)%dmc=8.5450511359999997
!      fwi_risk(i)%dc=19.013999999999999
!TEST ***************************

    IF ( fwi_risk(i)%dmc .LE.  0.4 * fwi_risk(i)%dc ) THEN
       uu =( 0.8 * fwi_risk(i)%dmc * fwi_risk(i)%dc ) / (fwi_risk(i)%dmc + 0.4 * fwi_risk(i)%dc)
    ELSE
       uu = fwi_risk(i)%dmc - (1.0 - 0.8 * fwi_risk(i)%dc / (fwi_risk(i)%dmc + 0.4 * fwi_risk(i)%dc )) * &
            &  (0.92 + (0.0114 * fwi_risk(i)%dmc)**(1.7))
    ENDIF
    fwi_risk(i)%bui=MAX(uu,0.0)

    IF (( fwi_risk(i)%dmc .EQ. 0. ) .AND. ( fwi_risk(i)%dc .EQ. 0. )) THEN
        fwi_risk(i)%bui=0.0
    ENDIF
!    WRITE (9,*) 'fwi_risk(i)%bui',fwi_risk(i)%bui,'8.4904265358371838 '
! 6  Fire Weather Index
!==========================================================================
!
!
!    '''Calculates today's Fire Weather Index
!Paramteres:
!    ISI is the current day's ISI
!    BUI is the current day's BUI'''
!test:
!TEST ***************************
! FWI should be  10.096371392382368 if
!    fwi_risk(i)%isi=10.853661073655068
!    fwi_risk(i)%bui=8.4904265358371838
!TEST ***************************
    IF (fwi_risk(i)%bui .LE. 80.0)THEN
       fD = 0.626 * fwi_risk(i)%bui**(0.809) + 2.0
    ELSE
       fD = 1000.0 / (25.0 + 108.64 *EXP(-0.023 *fwi_risk(i)%bui))
    ENDIF
    fwiB = 0.1 * fwi_risk(i)%isi * fD

    IF (fwiB .GT. 1.0) THEN
       fwi_risk(i)%fwi = MAX(EXP(2.72 *(0.434 * LOG(fwiB))**(0.647)),0.0)
    ELSE
       fwi_risk(i)%fwi = MAX(fwiB,0.0)
    ENDIF
!    WRITE (9,*) 'fwi_risk(i)%fwi',fwi_risk(i)%fwi,'10.096371392382368 '

! IF YOU WANT TO MASK FOR SNOW uncomment these three lines
   !    IF (zsnow .EQ. 1 ) THEN
   !        fwi_risk(i)%bui=0.0
   !        fwi_risk(i)%isi=0.0
   !        fwi_risk(i)%fwi=0.0
   !     END IF



    IF (fwi_risk(i)%fwi .LT. 5.2) THEN
        fwi_risk(i)%danger_risk=1.0
    ELSE IF (fwi_risk(i)%fwi .LT. 11.2) THEN
            fwi_risk(i)%danger_risk=2.0
    ELSE IF (fwi_risk(i)%fwi .LT. 21.3) THEN
        fwi_risk(i)%danger_risk=3.0
    ELSE IF (fwi_risk(i)%fwi .LT. 38.0) THEN
        fwi_risk(i)%danger_risk=4.0
    ELSE IF (fwi_risk(i)%fwi .LT. 50.0) THEN
        fwi_risk(i)%danger_risk=5.0
    ELSE
        fwi_risk(i)%danger_risk=6.0
    ENDIF

    fwi_risk(i)%dsr=0.0272*(fwi_risk(i)%fwi**(1.77))

    IF (ieee_is_nan(fwi_risk(i)%fwi)) THEN
        PRINT *, 'FWI: ', fwi_risk(i)%fwi
        PRINT *, 'ISI: ', fwi_risk(i)%isi        
        PRINT *, 'BUI: ', fwi_risk(i)%bui
        PRINT *, 'FFMC: ', fwi_risk(i)%ffmc
        PRINT *, 'DMC: ', fwi_risk(i)%dmc
        PRINT *, 'DC: ', fwi_risk(i)%dc   
    ENDIF
    IF (fwi_risk(i)%fwi .GT. 180.0) THEN
        PRINT *, 'FWI: ', fwi_risk(i)%fwi
        PRINT *, 'ISI: ', fwi_risk(i)%isi        
        PRINT *, 'BUI: ', fwi_risk(i)%bui
        PRINT *, 'FFMC: ', fwi_risk(i)%ffmc
        PRINT *, 'DMC: ', fwi_risk(i)%dmc
        PRINT *, 'DC: ', fwi_risk(i)%dc
        PRINT *, 'TEMP: ', ztemp
        PRINT *, 'RAIN: ', zrain
        PRINT *, 'RH: ', zrh
        PRINT *, 'WIND: ', zwspeed 
        PRINT *, 'LAT: ', zlat 
    ENDIF
 ELSE  ! not a valid point for calculation 
   
! force point to fill values. This is done to avoid inconsistencies in the  event the mask of the reastart file is not the same as the forcings. This will avoid to pass over invalid points. 
 
    !FWI
    fwi_risk(i)%fwi=rfillvalue        
    fwi_risk(i)%ffmc=rfillvalue       
    fwi_risk(i)%dmc=rfillvalue        
    fwi_risk(i)%dc=rfillvalue         
    fwi_risk(i)%isi=rfillvalue        
    fwi_risk(i)%bui=rfillvalue        
    fwi_risk(i)%dsr=rfillvalue        
    fwi_risk(i)%danger_risk=rfillvalue
 ENDIF !non-lake or sea point

ENDDO !npoints

!!D)    OUTPUT
!------------
    IF (LEN(TRIM(output_file)) > 0) THEN
        CALL io_write_results(istep)
    ENDIF

    IF (actualdate .EQ. restartdate .AND. LEN(TRIM(output_restart)) > 0) THEN
        CALL io_write_restart
        output_restart = ''
    ENDIF


ENDDO ! date loop


  PRINT *, 'integration finished'
END PROGRAM geff
