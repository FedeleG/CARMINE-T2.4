! (C) Copyright 1996- ECMWF.
!
! This software is licensed under the terms of the Apache Licence Version 2.0
! which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
! In applying this licence, ECMWF does not waive the privileges and immunities
! granted to it by virtue of its status as an intergovernmental organisation nor
! does it submit to any jurisdiction.

!> @brief Tunable constants for model
!> @author Di Giuseppe, F., ECMWF
!> @author Maciel, P., ECMWF
MODULE mo_constants

    IMPLICIT NONE

    REAL, PARAMETER :: reps=1.0e-8             !< small positive threshold number
    REAL, PARAMETER :: r0CtoK=273.15           !< freezing point [K]
    REAL, PARAMETER :: tokmhr=3.6              !< conversion from [m s^-1] to [km h^-1]
    
    INTEGER, PARAMETER :: ndayinyear=365

    REAL, PARAMETER :: rfillvalue=-9999.0 ! missing value
    INTEGER, PARAMETER :: ifillvalue=-9999 ! missing value

END MODULE

