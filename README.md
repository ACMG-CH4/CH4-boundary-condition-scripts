# CH4-boundary-condition-scripts
### Description
Scripts, originally developed by Lu Shen, to generate Tropomi smoothed fields for usage as scaled boundary conditions.

### Instructions for running on aws
1. Make sure you have run the GCClassic simulation for the time period you are creating boundary conditions for. The run directory can be found at /home/ubuntu/run_gc_13.3.3
2. Make any changes necessary to the `boundary_condition_config.yml` file. This will likely include changing the default dates.
3. cd into `Step1_convert_GC` and run `./template_archive.py`. This will create a number of pkl files in the `data_converted_BC` directory. Note: `make_run.sh` is used for parallelising the creation of BC files. This is not yet supported on aws until we set up sbatch.
4. cd into `Step2_regrid_fast` and run `./read_daily.py`. This will create a `Daily_CH4.nc`file in the `Step3_correct_background` directory.
5. cd into the `Step3_correct_background` directory and run `Rscript correct_final.R`. This will create the Bias corrections file for CH4: `Bias_4x5_dk_2_updated.nc`.
6. cd into `Step4_write_boundary` and run `./write_boundary.py`. If you have the `upload_to_s3` setting set to true in your config file, the created boundary conditions will be uploaded to s3 in the config-specified output bucket.

Note: because tropomi data is hosted in a different aws region you will incur costs for downloading tropomi data.