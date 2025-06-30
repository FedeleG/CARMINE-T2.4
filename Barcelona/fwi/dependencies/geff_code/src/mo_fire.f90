! (C) Copyright 1996- ECMWF.
!
! This software is licensed under the terms of the Apache Licence Version 2.0
! which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
! In applying this licence, ECMWF does not waive the privileges and immunities
! granted to it by virtue of its status as an intergovernmental organisation nor
! does it submit to any jurisdiction.

!> @brief Meteorological and fire fields/variables
!> @author Di Giuseppe, F., ECMWF
!> @author Maciel, P., ECMWF
MODULE mo_fire

    IMPLICIT NONE

    INTEGER :: ntimestep  !< total integration time
    INTEGER :: npoints    !< number of points

    REAL, PARAMETER :: stmspd=30         !< [mile h^-1] translational speed of a storm
    REAL, PARAMETER :: pnorm1=0.00232    !< needed in ignition component
    REAL, PARAMETER :: pnorm2=0.99767    !< needed in ignition component
    REAL, PARAMETER :: pnorm3=0.0000185  !< needed in ignition component

    ! fields/variables defined in time (SIZE() = ntimestep)
    INTEGER, ALLOCATABLE :: nhours(:)

    ! fields/variables defined in space (SIZE() = npoints)
    REAL, ALLOCATABLE :: lons(:)         !< [degree] longitude
    REAL, ALLOCATABLE :: lats(:)         !< [degree] latitude
    REAL, ALLOCATABLE :: rtemp(:)        !< [K] temperature
    REAL, ALLOCATABLE :: rrh(:)          !< [fraction] relative humidity
    REAL, ALLOCATABLE :: rrain(:)        !< [mm day^-1] rainfall/accumulated precipitation
    REAL, ALLOCATABLE :: rwspeed(:)      !< [m s^-1] wind speed
    REAL, ALLOCATABLE :: rlsm(:)         !< [fraction] land-sea mask (1=land, 0=sea, 0.5=coast)



END MODULE mo_fire
