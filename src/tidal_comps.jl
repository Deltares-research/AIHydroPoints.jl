# compute phases for the tidal constituents using Doodson numbers
# The code is based on the hatyan code by Deltares, which is a translation of the Fortran code by Schureman.
# see: https://github.com/Deltares/hatyan

using Dates

# default reference date for the doodson calculations
const default_refdate = DateTime(1900,1,1)

"""
    get_doodson_eqvals(dood_date::DateTime)

Computes doodson values for the given time.

# Arguments

- `dood_date::DateTime`: Date on which doodson value should be calculated.
"""
function get_doodson_eqvals(dood_date::DateTime)
    # Number of Julian centuries (36525 days) with respect to Greenwich mean noon, 31 December 1899 (Gregorian calendar)
    DNUJE = 24*36525
    dood_tstart_sec = robust_timedelta_sec(dood_date)
    dood_Tj = (dood_tstart_sec/3600+12)/(DNUJE) 

    # From hatyan documentation. See also p162 Schureman book
    dood_T_rad = deg2rad(180 + hour(dood_date)*15.0+minute(dood_date)*15.0/60)
    dood_S_rad =  (4.7200089 + 8399.7092745*dood_Tj + 0.0000346*dood_Tj^2)
    dood_H_rad =  (4.8816280 + 628.3319500*dood_Tj  + 0.0000052*dood_Tj^2)
    dood_P_rad =  (5.8351526 + 71.0180412*dood_Tj   - 0.0001801*dood_Tj^2)
    dood_N_rad =  (4.5236016 - 33.7571463*dood_Tj   + 0.0000363*dood_Tj^2)
    dood_P1_rad = (4.9082295 + 0.0300053*dood_Tj    + 0.0000079*dood_Tj^2)
    # from SLS table 4.2: pd.DataFrame(dict(const=[218.32,280.47,83.35,125.04,282.94],T=[481267.88,36000.77,4069.01,-1934.14,1.72]),index=['S','H','P','N','P1'])#/180*np.pi
    doodson=[dood_T_rad, dood_S_rad, dood_H_rad, dood_P_rad, dood_N_rad, dood_P1_rad] # ['T','S','H','P','N','P1']
    return doodson
end

function get_doodson_eqvals(dood_date::Vector{DateTime})
    doodson = zeros(length(dood_date),6)
    for i in eachindex(dood_date)
        doodson[i,:] .= get_doodson_eqvals(dood_date[i])
    end
    return doodson
end

"""
    robust_timedelta_sec(dood_date::DateTime; kwargs...)

Compute time in seconds relative to a reference.
Defaulf reference is January 1st 1900

# Arguments

- `dood_date::DateTime` 

# Keywords

- `refdate_dt::DateTime`: Reference date and time. By default January 1st 1900.
    (**Default**: `default_refdate`)
"""
function robust_timedelta_sec(dood_date::DateTime; refdate_dt::DateTime=default_refdate)
    return (dood_date - refdate_dt).value / 1000.0 # msec to sec
end

function robust_timedelta_sec(dood_date::Vector{DateTime}; refdate_dt::DateTime=default_refdate)
    return robust_timedelta_sec.(dood_date, refdate_dt)
end

"""
    lunar2solar(lunar_doodson::Vector{Float64})

Convert lunar doodson to solar doodson
Some tidal constituents are defined in lunar doodson numbers, but we need solar doodson numbers for the calculations.
List of lunar doodson numbers: L, S, H, P, N, P1 at 
https://iho.int/mtg_docs/com_wg/IHOTC/IHOTC_Misc/TWCWG_Constituent_list.pdf
and subset of solar doodson numbers: T, S, H, P, N, P1 at
https://github.com/Deltares/hatyan/blob/main/hatyan/data/data_schureman_harmonic.csv

# Arguments

- `lunar_doodson::Vector{Float64}`: Input lunar doodson numbers.
"""
function lunar2solar(lunar_doodson::Vector{Float64})
    # L -> T - S + H 
    L=lunar_doodson[1]
    S=lunar_doodson[2]
    H=lunar_doodson[3]
    P=lunar_doodson[4]
    N=lunar_doodson[5]
    P1=lunar_doodson[6]
    return [L, S-L, H+L, P, N, P1]
end

#
# Some tidal constituents
# https://iho.int/mtg_docs/com_wg/IHOTC/IHOTC_Misc/TWCWG_Constituent_list.pdf
# defined as doodson numbers 
# some o
constituents=Dict{String,Vector{Float64}}()
constituents["T"]=[1.0,0.0,0.0,0.0,0.0,0.0]
constituents["S"]=[0.0,1.0,0.0,0.0,0.0,0.0]
constituents["H"]=[0.0,0.0,1.0,0.0,0.0,0.0]
constituents["P"]=[0.0,0.0,0.0,1.0,0.0,0.0]
constituents["N"]=[0.0,0.0,0.0,0.0,1.0,0.0]
constituents["P1"]=[0.0,0.0,0.0,0.0,0.0,1.0]

# M2 is the principal lunar semidiurnal constituent
constituents["SSA"]=lunar2solar([0.0,0.0,2.0,0.0,0.0,0.0])

constituents["K1"]=lunar2solar([1.0,1.0,0.0,0.0,0.0,1.0])
constituents["O1"]=lunar2solar([1.0,-1.0,0.0,0.0,0.0,-1.0])
constituents["Q1"]=lunar2solar([1.0,-2.0,0.0,1.0,0.0,0.0])
constituents["P1"]=lunar2solar([1.0,1.0,-2.0,0.0,0.0,0.0]) #close to K1

constituents["M2"]=lunar2solar([2.0,0.0,0.0,0.0,0.0,0.0])
constituents["S2"]=lunar2solar([2.0,2.0,-2.0,0.0,0.0,0.0])
constituents["N2"]=lunar2solar([2.0,-1.0,0.0,1.0,0.0,0.0])
constituents["K2"]=lunar2solar([2.0,2.0,0.0,0.0,0.0,0.0]) #close to S2

"""
    primary_frequencies_as_doodson(freqs::Vector{String})

Gives solar doodson numbers of named pre-defined tidal constituents.

# Arguments

- `freqs::Vector{String}`: Array of named tidal constituents. 
"""
function primary_frequencies_as_doodson(freqs::Vector{String})
    values = zeros(6, length(freqs))
    for i in eachindex(freqs)
        values[:,i] .= constituents[freqs[i]]
    end
    return values    
end