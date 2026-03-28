# settings.jl — HatyanSettings struct for tidal analysis and prediction

"""
    HatyanSettings

Groups the numerical options that control tidal analysis and prediction.

Fields:
- `nodalfactors`: apply nodal factor corrections (f and u); default `true`
- `fu_alltimes`: compute f/u at every timestep rather than at period centre; default `true`
- `xfac`: apply x-factor amplitude correction to f; default `false`

The constituent table method (`"schureman"`) is **not** stored here; it is
passed as a separate `method` argument to `analysis` and embedded in the
`TidalConstituents.source` provenance string.
"""
struct HatyanSettings
    nodalfactors :: Bool
    fu_alltimes  :: Bool
    xfac         :: Bool
end

"""
    HatyanSettings(; nodalfactors=true, fu_alltimes=true, xfac=false)

Construct a `HatyanSettings` with keyword arguments, using the documented defaults.
"""
HatyanSettings(;
    nodalfactors :: Bool = true,
    fu_alltimes  :: Bool = true,
    xfac         :: Bool = false,
) = HatyanSettings(nodalfactors, fu_alltimes, xfac)
