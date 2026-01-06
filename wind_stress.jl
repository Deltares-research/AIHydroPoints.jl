# wind_stress.jl
# compute wind stress from 10meter wind
#
# This version is a first relatively simple versionusing a constant Chranock coefficient
# and a constant air density. The wind stress is computed as:
# tau = rho_air * Cd * |u| * u
# where rho_air is the air density, Cd is the Charnock coefficient, and u is the wind vector.
# Cd = ( u_star / u )^2
# The Chranock parametrization is defined as:
# z_0 = α_ch u_star^2 /g
# where z_0 is the roughness length, α_ch is the Charnock coefficient, u_star is the friction velocity, and g is the gravity.
# The friction velocity is defined as:
# u_star = sqrt( |tau| / rho_air )
# Finally, we assume a logaritmic wind profile:
# u = u_star / κ * log( z / z_0 )
# where κ is the von Karman constant.

# packages

# constants
const α_ch = 0.041 # Charnock coefficient
const κ = 0.4 # von Karman constant
const g = 9.81 # gravity
const ρ_air = 1.20 # air density
const Cd_init = 0.002 # initial guess for Cd
const z_ref = 10.0 # reference height 10 meters
const tol = 1e-6 # tolerance for Newton-Raphson
const max_iter = 10 # maximum number of iterations for Newton-Raphson

"""
 Compute wind from stress and its derivative. This is the inverse of what we want, so it's input to the Newton-Raphson iteration.

 function stress_to_wind_and_derivative(τ)
"""
function stress_to_wind_and_derivative(τ)
    u_star = sqrt( abs(τ) / ρ_air )
    z_0 = α_ch * u_star^2 / g
    u = u_star / κ * log( 10 / z_0 )
    dτ = one(τ) * sign(τ)
    du_star = dτ*0.5 / u_star / ρ_air
    dz_0 = du_star * 2 * α_ch * u_star / g
    du = ( du_star / κ * log( 10 / z_0 )) - ( dz_0 * u_star / κ / z_0  )
    return u,du
end

"""
    Compute stress from wind with a constant Charnock coefficient and air density.
function wind_to_stress(u)
"""
function wind_to_stress(u)
    # first guess with constant Cd
    τ = ρ_air * Cd_init * u * u
    if u<0.01 # avoid division by zero, and initial guess is good enough
        return τ
    end
    # Newton-Raphson iteration
    for i in 1:max_iter
        u10,du10 = stress_to_wind_and_derivative(τ)
        Δτ = max(- (u10 - u) / du10, -0.8*τ)
        τ_new = τ + Δτ
        if abs(τ_new - τ) < tol
            break
        end
        τ = τ_new
    end
    return τ
end

function uv_to_stress_xy(u,v)
    u_mag = sqrt(u^2 + v^2)
    e_x= u/(u_mag+1e-6)
    e_y= v/(u_mag+1e-6)
    τ_mag = wind_to_stress(u_mag)
    return τ_mag*e_x, τ_mag*e_y
end