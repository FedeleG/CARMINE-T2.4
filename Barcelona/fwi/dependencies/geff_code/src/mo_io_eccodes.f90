! (C) Copyright 1996- ECMWF.
!
! This software is licensed under the terms of the Apache Licence Version 2.0
! which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
! In applying this licence, ECMWF does not waive the privileges and immunities
! granted to it by virtue of its status as an intergovernmental organisation nor
! does it submit to any jurisdiction.

!> @brief ecCodes GRIB I/O
!> @author Di Giuseppe, F., ECMWF
!> @author Maciel, P., ECMWF
MODULE mo_io_eccodes

    USE eccodes

    USE mo_control
    USE mo_fire
    USE mo_fwi
    USE mo_mark5
    USE mo_nfdrs
    USE mo_utilities

    IMPLICIT NONE
    PRIVATE

    PUBLIC :: io_getdata
    PUBLIC :: io_initialize
    PUBLIC :: io_write_restart
    PUBLIC :: io_write_results

    ! GRIB paramIds
    INTEGER, PARAMETER :: ilsm_pids(1)      = [172]
    INTEGER, PARAMETER :: irain_pids(1)     = [228]
    INTEGER, PARAMETER :: itemp_pids(1)     = [167]
    INTEGER, PARAMETER :: irh_pids(1)       = [168]
    INTEGER, PARAMETER :: iwspeed_pids(2)   = [165, 166]

    INTEGER, PARAMETER :: ifwi_risk_fwi_pids(1)         = [260540]
    INTEGER, PARAMETER :: ifwi_risk_ffmc_pids(1)        = [260541]
    INTEGER, PARAMETER :: ifwi_risk_dmc_pids(1)         = [260542]
    INTEGER, PARAMETER :: ifwi_risk_dc_pids(1)          = [260543]

    INTEGER, PARAMETER :: ifwi_risk_isi_pids(1)         = [260544]
    INTEGER, PARAMETER :: ifwi_risk_bui_pids(1)         = [260545]
    INTEGER, PARAMETER :: ifwi_risk_dsr_pids(1)         = [260546]
    INTEGER, PARAMETER :: ifwi_risk_danger_risk_pids(1) = [212027]

    TYPE :: GribField
        INTEGER :: fd       = 0
        INTEGER :: handle   = 0
        INTEGER :: paramId  = 0
        CHARACTER(LEN=20) :: shortName = ''
        CHARACTER(LEN=200) :: name     = ''
        INTEGER :: npoints  = 0
        INTEGER :: count    = 0
    CONTAINS
        PROCEDURE, PUBLIC :: open_as_input => gribfield_open_as_input
        PROCEDURE, PUBLIC :: open_as_output => gribfield_open_as_output
        PROCEDURE, PUBLIC :: open_as_restart => gribfield_open_as_restart
        PROCEDURE, PUBLIC :: next => gribfield_next
        PROCEDURE, PUBLIC :: close => gribfield_close
        PROCEDURE, PUBLIC :: coordinates => gribfield_coordinates
        PROCEDURE, PUBLIC :: values => gribfield_values
        PROCEDURE, PUBLIC :: values_as_integer => gribfield_values_as_integer
        PROCEDURE, PUBLIC :: header => gribfield_header
        PROCEDURE, PUBLIC :: same_geometry => gribfield_same_geometry
    END TYPE

    ! missing value indicator
    REAL, PARAMETER :: missingValue = rfillvalue !FIXME: -1.e20

    TYPE(GribField), TARGET :: input(6)

    TYPE(GribField), POINTER :: grib_lsm      => input(1)
    TYPE(GribField), POINTER :: grib_rain     => input(2)
    TYPE(GribField), POINTER :: grib_temp     => input(3)
    TYPE(GribField), POINTER :: grib_rh       => input(4)
    TYPE(GribField), POINTER :: grib_wspeed   => input(5)


CONTAINS


    SUBROUTINE io_getdata(istep)
        INTEGER, INTENT(IN) :: istep
        REAL, ALLOCATABLE :: tmp(:)

        CALL assert(npoints > 0)
        IF (istep == 1) RETURN

        IF (grib_rain%count    > 1) CALL next_values('io_getdata: rain', grib_rain, grib_rain%paramId, rrain)
        IF (grib_temp%count    > 1) CALL next_values('io_getdata: temp', grib_temp, grib_temp%paramId, rtemp)
        IF (grib_rh%count      > 1) CALL next_values('io_getdata: rh', grib_rh, grib_rh%paramId, rrh)
        IF (grib_wspeed%count  > 1) CALL next_values('io_getdata: wspeed', grib_wspeed, grib_wspeed%paramId, rwspeed)

    END SUBROUTINE


    SUBROUTINE next_values(message, grib, paramId, values)
        CHARACTER(LEN=*), INTENT(IN) :: message
        TYPE(GribField), INTENT(INOUT) :: grib
        INTEGER, INTENT(IN) :: paramId
        REAL, INTENT(INOUT), ALLOCATABLE :: values(:)
        CALL assert(SIZE(values) == npoints, message//' SIZE(values) == npoints')

        CALL assert(grib%next(), message//' grib%next()')
        CALL grib%header()

        CALL assert(grib%npoints == npoints, message//' grib%npoints == npoints')
        CALL assert(grib%paramId == paramId, message//' grib%paramId == paramId')

        CALL grib%values(values)
    END SUBROUTINE


    SUBROUTINE io_initialize
        TYPE(GribField) :: restart
        INTEGER :: i
        REAL, ALLOCATABLE :: tmp(:)

        ! use land-sea mask to define geometry
        CALL grib_lsm%open_as_input(lsmfile, 'land-sea mask', ilsm_pids)

        npoints = grib_lsm%npoints
        PRINT *, 'Number of points: ', npoints

        CALL assert(npoints > 0)
        ALLOCATE(lats(npoints))
        ALLOCATE(lons(npoints))

        CALL assert(grib_lsm%next(), 'io_initialize: grib_lsm%next()')
        CALL grib_lsm%coordinates(lats, lons)

        CALL grib_temp%open_as_input(tempfile, 'temperature', itemp_pids)
        CALL grib_rh%open_as_input(rhfile, 'relative humidity', irh_pids)
        CALL grib_rain%open_as_input(rainfile, 'rainfall', irain_pids)
        CALL grib_wspeed%open_as_input(wspeedfile, 'wind speed', iwspeed_pids)

        ! fields/variables defined in time (SIZE() = ntimestep)
        ntimestep = MAXVAL(input(:)%count)
        PRINT *, 'Number of time steps: ', ntimestep

        CALL assert(ntimestep > 0, "io_initialize: ntimestep > 0")
        ALLOCATE(nhours(ntimestep))
        DO i = 1, ntimestep
            nhours(i) = (i - 1) * dt
        ENDDO

        ! fields/variables defined in space (SIZE() = npoints)
        CALL assert(npoints > 0, "io_initialize: npoints > 0")
        ALLOCATE(tmp(npoints))

        DO i = 1, SIZE(input)
            IF (i > 1) CALL assert(input(i)%same_geometry(input(1)))
            CALL assert(input(i)%count == 1 .OR. input(i)%count == ntimestep)
        ENDDO

        ALLOCATE(rlsm(npoints))
        ! CALL assert(grib_lsm%next())  ! already called (above)
        CALL grib_lsm%values(rlsm)

        ALLOCATE(rrain(npoints))
        CALL next_values('io_initialize: grib_rain', grib_rain, grib_rain%paramId, rrain)

        ALLOCATE(rtemp(npoints))
        CALL next_values('io_initialize: grib_temp', grib_temp, grib_temp%paramId, rtemp)

        ALLOCATE(rrh(npoints))
        CALL next_values('io_initialize: grib_rh', grib_rh, grib_rh%paramId, rrh)

        ALLOCATE(rwspeed(npoints))
        CALL next_values('io_initialize: grib_wspeed', grib_wspeed, grib_wspeed%paramId, rwspeed)

        ALLOCATE(fwi_risk(npoints))

        IF (LEN(TRIM(restart_file)) > 0) THEN
            PRINT *, "Initialization type: exact initialization from '" // TRIM(restart_file) // "'"

            CALL restart%open_as_restart(restart_file)

            CALL next_values('restart fwi_risk%ffmc', restart, ifwi_risk_ffmc_pids(1), tmp)
            fwi_risk(:)%ffmc = tmp

            CALL next_values('restart fwi_risk%dmc', restart, ifwi_risk_dmc_pids(1), tmp)
            fwi_risk(:)%dmc = tmp

            CALL next_values('restart fwi_risk%dc', restart, ifwi_risk_dc_pids(1), tmp)
            fwi_risk(:)%dc = tmp

            CALL restart%close()
        ELSE
            PRINT *, 'Initialization type: artificial conditions'

            ! Here the loop is necessary
            ! Dead fuel
            DO i = 1, npoints
                IF (rlsm(i) .GT. 0.0001 ) THEN

                    ! For the FWI moisture code values (FFMC=85, DMC=6, DC=15) provide a
                    ! reasonable set of conditions for post-snowmelt springtime conditions
                    ! in eastern/central Canada, the Northern U.S., and Alaska; physically
                    ! these spring start-up values represent about 3 days of drying from
                    ! complete moisture saturation of the fuel layer. In areas or years
                    ! with particularly dry winters (or parts of the world without
                    ! significant snow cover) these start-up values for FFMC and DMC may
                    ! still be appropriate as these two elements respond relatively quickly
                    ! to changes in the weather. The DC component however, because of its
                    ! very long response time, can take considerable time to adjust to
                    ! unrealistic initial values and some effort to estimate over-winter
                    ! value of the DC may be necessary. Users can look again to Lawson and
                    ! Armitage (2008) for a more detailed description of code calculation
                    ! startup issues and the over-winter adjustment process.
                    fwi_risk(i)%ffmc = 85.
                    fwi_risk(i)%dmc  =  6.
                    fwi_risk(i)%dc   = 15.
                ENDIF
            ENDDO
        ENDIF

        DEALLOCATE(tmp)
    END SUBROUTINE

    SUBROUTINE io_write_restart
        INTEGER :: fd

        ! Open restart file
        fd = 0
        CALL assert(LEN(TRIM(output_restart)) > 0, 'output_restart not empty')
        CALL codes_open_file(fd, output_restart, 'w')
        CALL assert(fd /= 0, 'codes_open_file (w): '//TRIM(output_restart))

        CALL write_field(fd, ifwi_risk_ffmc_pids(1), fwi_risk(:)%ffmc)
        CALL write_field(fd, ifwi_risk_dmc_pids(1), fwi_risk(:)%dmc)
        CALL write_field(fd, ifwi_risk_dc_pids(1), fwi_risk(:)%dc)

        CALL codes_close_file(fd)
    END SUBROUTINE

    SUBROUTINE io_write_results(istep)
        INTEGER, INTENT(IN) :: istep  ! NOTE: ignored
        INTEGER :: fd
        CHARACTER, SAVE :: cmode = 'w'

        ! Open results file
        fd = 0
        CALL assert(LEN(TRIM(output_file)) > 0, 'output_file not empty')
        CALL codes_open_file(fd, output_file, cmode)
        CALL assert(fd /= 0, 'codes_open_file ('//cmode//'): '//TRIM(output_file))
        cmode = 'a'

        CALL write_field(fd, irain_pids(1), rrain)
        CALL write_field(fd, itemp_pids(1), rtemp)
        CALL write_field(fd, irh_pids(1), rrh)
        CALL write_field(fd, iwspeed_pids(1), rwspeed)

        CALL write_field(fd, ifwi_risk_fwi_pids(1), fwi_risk(:)%fwi)
        CALL write_field(fd, ifwi_risk_ffmc_pids(1), fwi_risk(:)%ffmc)
        CALL write_field(fd, ifwi_risk_dmc_pids(1), fwi_risk(:)%dmc)
        CALL write_field(fd, ifwi_risk_dc_pids(1), fwi_risk(:)%dc)

        CALL write_field(fd, ifwi_risk_isi_pids(1), fwi_risk(:)%isi)
        CALL write_field(fd, ifwi_risk_bui_pids(1), fwi_risk(:)%bui)
        CALL write_field(fd, ifwi_risk_dsr_pids(1), fwi_risk(:)%dsr)
        CALL write_field(fd, ifwi_risk_danger_risk_pids(1), fwi_risk(:)%danger_risk)

        IF (output_constant) THEN
            CALL write_field(fd, ilsm_pids(1), rlsm)
        ENDIF

        CALL codes_close_file(fd)
    END SUBROUTINE

    SUBROUTINE gribfield_coordinates(this, latitudes, longitudes)
        CLASS(GribField), INTENT(IN) :: this
        REAL, ALLOCATABLE, INTENT(INOUT) :: latitudes(:)
        REAL, ALLOCATABLE, INTENT(INOUT) :: longitudes(:)
        REAL, ALLOCATABLE :: tmp(:)

        CALL assert(ALLOCATED(latitudes) .AND. SIZE(latitudes) == this%npoints, 'latitudes allocation')
        CALL assert(ALLOCATED(longitudes) .AND. SIZE(longitudes) == this%npoints, 'longitudes allocation')
        latitudes = 0
        longitudes = 0

        ALLOCATE(tmp(this%npoints))
        CALL codes_grib_get_data(this%handle, latitudes, longitudes, tmp)
        DEALLOCATE(tmp)
    END SUBROUTINE

    SUBROUTINE gribfield_values(this, values)
        CLASS(GribField), INTENT(IN) :: this
        REAL, ALLOCATABLE, INTENT(INOUT) :: values(:)
        INTEGER :: n, bitmapPresent

        CALL assert(ALLOCATED(values), 'gribfield_values values allocated')
        CALL assert(SIZE(values) == this%npoints, 'gribfield_values: values size mismatch)')

        CALL codes_get_size(this%handle, "values", n)
        CALL assert(SIZE(values) == n, 'gribfield_values: codes_get_size("values") mismatch')

        ! ensure all missing values use the same indicator ('set' before 'get')
        CALL codes_get(this%handle, "bitmapPresent", bitmapPresent)
        IF (bitmapPresent /= 0) CALL codes_set(this%handle, "missingValue", missingValue)

        CALL codes_get(this%handle, "values", values)
    END SUBROUTINE

    SUBROUTINE gribfield_values_as_integer(this, values)
        CLASS(GribField), INTENT(IN) :: this
        INTEGER, ALLOCATABLE, INTENT(INOUT) :: values(:)
        REAL, ALLOCATABLE :: rvalues(:)
        CALL assert(ALLOCATED(values), 'gribfield_values_as_integer: values allocated')

        ALLOCATE(rvalues(SIZE(values)))
        CALL gribfield_values(this, rvalues)

        values = rvalues
        DEALLOCATE(rvalues)
    END SUBROUTINE

    SUBROUTINE gribfield_header(this)
        CLASS(GribField), INTENT(INOUT) :: this
        CALL assert(this%handle /= 0, 'this%handle')

        CALL codes_get(this%handle, 'shortName', this%shortName)
        CALL codes_get(this%handle, 'name', this%name)
        CALL codes_get(this%handle, 'paramId', this%paramid)
        ! CALL assert(this%paramId > 0, 'paramId > 0')

        CALL codes_get(this%handle, 'numberOfDataPoints', this%npoints)
        CALL assert(this%npoints > 0, 'numberOfDataPoints > 0')
    END SUBROUTINE

    FUNCTION gribfield_next(this) RESULT(r)
        CLASS(GribField), INTENT(INOUT) :: this
        INTEGER :: iret
        LOGICAL :: r
        IF (this%handle /= 0) CALL codes_release(this%handle)
        CALL codes_grib_new_from_file(this%fd, this%handle, iret)
        r = iret /= CODES_END_OF_FILE
        CALL assert(.NOT. r .OR. iret == 0 .AND. this%handle /= 0, 'codes_grib_new_from_file')
    END FUNCTION

    SUBROUTINE gribfield_close(this)
        CLASS(GribField), INTENT(INOUT) :: this
        IF (this%handle /= 0) CALL codes_release(this%handle)
        this%handle = 0
        CALL codes_close_file(this%fd)
        this%fd = 0
    END SUBROUTINE

    SUBROUTINE gribfield_open_as_input(this, file, var, pids)
        CLASS(GribField), INTENT(INOUT) :: this

        CHARACTER(LEN=*), INTENT(IN) :: file
        CHARACTER(LEN=*), INTENT(IN) :: var
        INTEGER, DIMENSION(:), INTENT(IN) :: pids

        INTEGER :: i, n
        LOGICAL :: found

        ! open file and read messages to the end
        CALL codes_open_file(this%fd, file, 'r')
        CALL assert(this%next(), 'file "'//TRIM(file)//'": '//TRIM(var)//' GRIB not found')

        ! first GRIB message: get header (variable name/id and geometry), then
        ! next GRIB messages: confirm numberOfDataPoints, increment count
        CALL this%header()

        found = .FALSE.
        DO i = 1, SIZE(pids)
            found = this%paramId .EQ. pids(i)
            IF (found) EXIT
        ENDDO
        CALL assert(found, 'file "'//TRIM(file)//'": '//TRIM(var)//' field not found')
        PRINT *, 'Found '//TRIM(var)//' field with paramId=', this%paramId

        this%count = 1
        DO WHILE (this%next())
            this%count = this%count + 1
            CALL codes_get(this%handle, 'paramId', i)
            CALL codes_get(this%handle, 'numberOfDataPoints', n)
            CALL assert(n == this%npoints .AND. i == this%paramId,&
            'Input fields should have the same paramId and geometry (numberOfDataPoints)')
        ENDDO

        ! end reached, re-open the file to read messages one-by-one
        CALL this%close()
        CALL codes_open_file(this%fd, file, 'r')
        CALL assert(this%fd /= 0, 'file "'//TRIM(file)//'": GRIB codes_open_file (r)')
    END SUBROUTINE

    SUBROUTINE gribfield_open_as_output(this, file)
        CLASS(GribField), INTENT(INOUT) :: this
        CHARACTER(LEN=*), INTENT(IN) :: file

        this%fd = 0
        this%handle = 0
        CALL assert(LEN(file) > 0, 'file "'//TRIM(file)//'": invalid file name')
        CALL codes_open_file(this%fd, file, 'a')
        CALL assert(this%fd /= 0, 'file "'//TRIM(file)//'": GRIB codes_open_file (a)')
    END SUBROUTINE

    SUBROUTINE gribfield_open_as_restart(this, file)
        CLASS(GribField), INTENT(INOUT) :: this
        CHARACTER(LEN=*), INTENT(IN) :: file

        this%fd = 0
        this%handle = 0
        CALL assert(LEN(file) > 0, 'file "'//TRIM(file)//'": invalid file name')
        CALL codes_open_file(this%fd, file, 'r')
        CALL assert(this%fd /= 0, 'file "'//TRIM(file)//'": GRIB codes_open_file (r)')
    END SUBROUTINE

    SUBROUTINE write_field(fd, paramid, values)
        INTEGER, INTENT(IN) :: fd, paramid
        REAL, INTENT(IN) :: values(:)
        INTEGER :: handle, i, bitmapPresent, edition
        TYPE(GribField), POINTER, SAVE :: ref => input(6)

        ! reference defines metadata (date/time/step/..., aside from paramId)
        IF (.NOT. (ref%count > 1)) THEN
            DO i = 2, SIZE(input)
                IF (input(i)%count > 1) THEN
                    ref => input(i)
                    EXIT
                ENDIF
            ENDDO
        ENDIF
        CALL assert(ref%handle /= 0)

        CALL assert(fd /= 0 .AND. paramid > 0, 'write_field: requirements')
        CALL assert(ref%npoints == SIZE(values), 'write_field: values size')

        handle = 0
        CALL codes_clone(ref%handle, handle)
        CALL assert(handle /= 0, 'write_field: codes_clone')

        ! FIXME fix GRIB2-only fields from GRIB1
        IF (paramid > 260000) THEN
            CALL codes_get(handle, 'edition', edition)
            IF (edition == 1 ) CALL codes_set(handle, 'edition', 2)
        ENDIF

        CALL codes_set(handle, 'paramId', paramid)
        CALL codes_set(handle, 'missingValue', missingValue)

        bitmapPresent = 0
        IF (ANY(values == missingValue)) bitmapPresent = 1
        CALL codes_set(handle, 'bitmapPresent', bitmapPresent)

        CALL codes_set(handle, 'values', values)

        CALL codes_write(handle, fd)
        CALL codes_release(handle)
    END SUBROUTINE

    SUBROUTINE write_field_from_integer(fd, paramid, values)
        INTEGER, INTENT(IN) :: fd, paramid
        INTEGER, INTENT(IN) :: values(:)
        REAL, ALLOCATABLE :: tmp(:)

        ALLOCATE(tmp(SIZE(values)))
        tmp = values
        CALL write_field(fd, paramid, tmp)
        DEALLOCATE(tmp)
    END SUBROUTINE

    FUNCTION gribfield_same_geometry(this, other) RESULT(yes)
        CLASS(GribField), INTENT(INOUT) :: this, other
        LOGICAL :: yes
        yes = this%npoints == other%npoints
        yes = yes .AND. (this%count == 1 .OR. other%count == 1 .OR. this%count == other%count)
        IF (.not. yes) THEN
            PRINT *, "%ERROR: fields don't have the same geometry: ",&
            char(10), "(paramId=", this%paramId, ", numberOfDataPoints=", this%npoints, ", count=", this%count,")",&
            char(10), "(paramId=", other%paramId, ", numberOfDataPoints=", other%npoints, ", count=", other%count,")"
            STOP 1
        ENDIF
    END FUNCTION

END MODULE
