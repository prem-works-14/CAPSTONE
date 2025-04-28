import xarray as xr

ds = xr.open_dataset("C:/Users/Dell/Downloads/climax_training_setup/data/sample/era5.nc")
print(ds)