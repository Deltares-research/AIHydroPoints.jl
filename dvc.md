

## Installation

We use pixi here for installation. This add it to the `pixi.toml` file and we need to use it as `pixi run dvc`. 

## First use for repo

- see [dvc getting started](https://doc.dvc.org/start)
- `pixi add dvc`
- `pixi add dvc-s3`
- `pixi run dvc init` initialize dvc inside repo
- commit the changes to git
- `pixi run dvc remote add minio_deltares_ai-hydro -d s3://ai-hydro/dvc`
- `pixi run dvc remote modify minio_deltares_ai-hydro endpointurl https://s3.deltares.nl`
- `pixi run dvc remote modify --local minio_deltares_ai-hydro access_key_id ACCESS_KEY`
- `pixi run dvc remote modify --local minio_deltares_ai-hydro secret_access_key SECRET_ACCESS_KEY`

## First use after new clone
- restore `.dvc/config.local
    - `pixi run dvc remote modify --local minio_deltares_ai-hydro access_key_id ACCESS_KEY`
    - `pixi run dvc remote modify --local minio_deltares_ai-hydro secret_access_key SECRET_ACCESS_KEY`

## Add a file
- `pixi run add FILENAME`
- add the dvc file that appears to git
- `pixi run dvc push`

## Downloading files
- `pixi run dvc pull FILENAME`
    The filename is optional and without it all files will be downloaded.
