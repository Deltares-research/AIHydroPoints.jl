using TOML

"""
    toml_write(path::String, dict::Dict; overwrite::Bool=false)

Write `dict` to a TOML file at `path`.

Throws an error if the parent directory does not exist, or if the file already
exists and `overwrite` is `false`.
"""
function toml_write(path::String, dict::Dict; overwrite::Bool=false)
    isdir(dirname(path)) || error("directory does not exist: $(dirname(path))")
    !overwrite && isfile(path) && error("file already exists (use overwrite=true): $path")
    open(path, "w") do io
        TOML.print(io, dict)
    end
end

"""
    toml_read(path::String) -> Dict{String,Any}

Parse a TOML file and return its contents as a nested `Dict{String,Any}`.
"""
function toml_read(path::String)
    isfile(path) || error("TOML file not found: $path")
    return TOML.parsefile(path)
end
